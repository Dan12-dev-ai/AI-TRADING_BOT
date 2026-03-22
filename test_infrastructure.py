"""
Test suite for Infrastructure Module
Production-ready tests for Docker, Kubernetes, and Vault configurations
"""

import pytest
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List


class TestDockerConfiguration:
    """Test cases for Docker configuration"""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is properly structured"""
        dockerfile_path = Path(__file__).parent.parent / "docker" / "deployment" / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile should exist"
        
        content = dockerfile_path.read_text()
        
        # Check for required stages
        assert "FROM python:3.12-slim as builder" in content
        assert "FROM python:3.12-slim as production" in content
        
        # Check for security best practices
        assert "RUN groupadd -r medallionx" in content
        assert "USER medallionx" in content
        
        # Check for health check
        assert "HEALTHCHECK" in content
        
        # Check for proper environment setup
        assert "ENV PATH=" in content
        assert "ENV PYTHONUNBUFFERED=1" in content

    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists and is valid"""
        compose_path = Path(__file__).parent.parent / "docker" / "deployment" / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml should exist"
        
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check required services
        required_services = ['redis', 'postgres', 'dashboard', 'trading-engine', 'nginx']
        for service in required_services:
            assert service in compose_config['services'], f"Service {service} should be defined"
        
        # Check Redis configuration
        redis_config = compose_config['services']['redis']
        assert 'volumes' in redis_config
        assert 'healthcheck' in redis_config
        assert 'restart' in redis_config
        
        # Check PostgreSQL configuration
        postgres_config = compose_config['services']['postgres']
        assert 'environment' in postgres_config
        assert 'POSTGRES_PASSWORD' in str(postgres_config['environment'])
        assert 'volumes' in postgres_config
        assert 'healthcheck' in postgres_config
        
        # Check trading engine configuration
        trading_config = compose_config['services']['trading-engine']
        assert 'environment' in trading_config
        assert 'depends_on' in trading_config
        assert 'deploy' in trading_config
        assert 'resources' in trading_config['deploy']
        
        # Check network configuration
        assert 'networks' in compose_config
        assert 'medallionx-network' in compose_config['networks']
        
        # Check volume configuration
        assert 'volumes' in compose_config
        required_volumes = ['redis_data', 'postgres_data', 'prometheus_data', 'grafana_data']
        for volume in required_volumes:
            assert volume in compose_config['volumes']

    def test_docker_compose_security(self):
        """Test Docker Compose security configuration"""
        compose_path = Path(__file__).parent.parent / "docker" / "deployment" / "docker-compose.yml"
        
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check that no services run as root (except where necessary)
        for service_name, service_config in compose_config['services'].items():
            if service_name not in ['nginx', 'vault']:  # Services that might need root
                # Check for user directive or non-root user
                if 'user' in service_config:
                    assert service_config['user'] != 'root:root', f"Service {service_name} should not run as root"
        
        # Check for proper resource limits
        critical_services = ['trading-engine', 'dashboard', 'postgres', 'redis']
        for service in critical_services:
            if service in compose_config['services']:
                service_config = compose_config['services'][service]
                if 'deploy' in service_config and 'resources' in service_config['deploy']:
                    resources = service_config['deploy']['resources']
                    assert 'limits' in resources, f"Service {service} should have resource limits"

    def test_docker_compose_monitoring(self):
        """Test Docker Compose monitoring configuration"""
        compose_path = Path(__file__).parent.parent / "docker" / "deployment" / "docker-compose.yml"
        
        with open(compose_path, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check monitoring services
        monitoring_services = ['prometheus', 'grafana', 'node-exporter']
        for service in monitoring_services:
            assert service in compose_config['services'], f"Monitoring service {service} should be defined"
        
        # Check Prometheus configuration
        prometheus_config = compose_config['services']['prometheus']
        assert 'volumes' in prometheus_config
        assert 'command' in prometheus_config
        assert '--storage.tsdb.retention.time=30d' in str(prometheus_config['command'])
        
        # Check Grafana configuration
        grafana_config = compose_config['services']['grafana']
        assert 'environment' in grafana_config
        assert 'GF_SECURITY_ADMIN_PASSWORD' in str(grafana_config['environment'])
        assert 'volumes' in grafana_config


class TestKubernetesConfiguration:
    """Test cases for Kubernetes configuration"""
    
    def test_deployment_yaml_exists(self):
        """Test that deployment.yaml exists and is valid"""
        deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        assert deploy_path.exists(), "deployment.yaml should exist"
        
        with open(deploy_path, 'r') as f:
            deploy_config = yaml.safe_load_all(f)
        
        # Find main deployment
        deployment = None
        for resource in deploy_config:
            if resource.get('kind') == 'Deployment' and 'trading-engine' in resource.get('metadata', {}).get('name', ''):
                deployment = resource
                break
        
        assert deployment is not None, "Trading engine deployment should be defined"
        
        # Check deployment configuration
        spec = deployment['spec']
        assert spec['replicas'] >= 3, "Should have at least 3 replicas for high availability"
        assert 'strategy' in spec, "Should have deployment strategy"
        assert spec['strategy']['type'] == 'RollingUpdate', "Should use rolling update strategy"
        
        # Check pod template
        pod_template = spec['template']
        assert 'securityContext' in pod_template['spec'], "Should have security context"
        assert pod_template['spec']['securityContext']['runAsNonRoot'] == True, "Should run as non-root"
        
        # Check container configuration
        containers = pod_template['spec']['containers']
        assert len(containers) > 0, "Should have at least one container"
        
        container = containers[0]
        assert 'resources' in container, "Should have resource limits"
        assert 'requests' in container['resources'], "Should have resource requests"
        assert 'limits' in container['resources'], "Should have resource limits"
        assert 'livenessProbe' in container, "Should have liveness probe"
        assert 'readinessProbe' in container, "Should have readiness probe"

    def test_kubernetes_monitoring_yaml_exists(self):
        """Test that monitoring.yaml exists and is valid"""
        monitoring_path = Path(__file__).parent.parent / "kubernetes" / "monitoring.yaml"
        assert monitoring_path.exists(), "monitoring.yaml should exist"
        
        with open(monitoring_path, 'r') as f:
            monitoring_config = yaml.safe_load_all(f)
        
        # Check for required monitoring components
        required_components = ['Deployment', 'Service', 'ConfigMap', 'Secret']
        found_components = set()
        
        for resource in monitoring_config:
            if resource and 'kind' in resource:
                found_components.add(resource['kind'])
        
        for component in required_components:
            assert component in found_components, f"Monitoring component {component} should be defined"
        
        # Check Prometheus deployment
        prometheus_deployment = None
        for resource in monitoring_config:
            if (resource and resource.get('kind') == 'Deployment' and 
                'prometheus' in resource.get('metadata', {}).get('name', '')):
                prometheus_deployment = resource
                break
        
        assert prometheus_deployment is not None, "Prometheus deployment should be defined"
        
        # Check Grafana deployment
        grafana_deployment = None
        for resource in monitoring_config:
            if (resource and resource.get('kind') == 'Deployment' and 
                'grafana' in resource.get('metadata', {}).get('name', '')):
                grafana_deployment = resource
                break
        
        assert grafana_deployment is not None, "Grafana deployment should be defined"

    def test_kubernetes_vault_yaml_exists(self):
        """Test that vault.yaml exists and is valid"""
        vault_path = Path(__file__).parent.parent / "kubernetes" / "vault.yaml"
        assert vault_path.exists(), "vault.yaml should exist"
        
        with open(vault_path, 'r') as f:
            vault_config = yaml.safe_load_all(f)
        
        # Check for required Vault components
        required_components = ['Deployment', 'Service', 'ConfigMap', 'Secret', 'Job']
        found_components = set()
        
        for resource in vault_config:
            if resource and 'kind' in resource:
                found_components.add(resource['kind'])
        
        for component in required_components:
            assert component in found_components, f"Vault component {component} should be defined"
        
        # Check Vault deployment
        vault_deployment = None
        for resource in vault_config:
            if (resource and resource.get('kind') == 'Deployment' and 
                resource.get('metadata', {}).get('name') == 'vault'):
                vault_deployment = resource
                break
        
        assert vault_deployment is not None, "Vault deployment should be defined"
        
        # Check Vault configuration
        spec = vault_deployment['spec']
        assert spec['replicas'] == 1, "Vault should have single replica for HA"
        assert 'securityContext' in spec['template']['spec'], "Should have security context"
        
        # Check for TLS configuration
        container = spec['template']['spec']['containers'][0]
        assert 'volumeMounts' in container, "Should have volume mounts"
        tls_mount = None
        for mount in container['volumeMounts']:
            if 'tls' in mount['name']:
                tls_mount = mount
                break
        assert tls_mount is not None, "Should mount TLS certificates"

    def test_kubernetes_rbac_configuration(self):
        """Test Kubernetes RBAC configuration"""
        deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        
        with open(deploy_path, 'r') as f:
            deploy_config = yaml.safe_load_all(f)
        
        # Check for RBAC components
        rbac_components = ['ServiceAccount', 'Role', 'RoleBinding']
        found_rbac = set()
        
        for resource in deploy_config:
            if resource and 'kind' in resource:
                if resource['kind'] in rbac_components:
                    found_rbac.add(resource['kind'])
        
        for component in rbac_components:
            assert component in found_rbac, f"RBAC component {component} should be defined"
        
        # Check role permissions
        role = None
        for resource in deploy_config:
            if (resource and resource.get('kind') == 'Role' and 
                resource.get('metadata', {}).get('name') == 'medallionx-role'):
                role = resource
                break
        
        assert role is not None, "Medallion-X role should be defined"
        assert 'rules' in role, "Role should have rules defined"
        assert len(role['rules']) > 0, "Role should have at least one rule"

    def test_kubernetes_network_policies(self):
        """Test Kubernetes network policies"""
        deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        
        with open(deploy_path, 'r') as f:
            deploy_config = yaml.safe_load_all(f)
        
        # Check for NetworkPolicy
        network_policy = None
        for resource in deploy_config:
            if (resource and resource.get('kind') == 'NetworkPolicy' and 
                resource.get('metadata', {}).get('name') == 'medallionx-network-policy'):
                network_policy = resource
                break
        
        assert network_policy is not None, "Network policy should be defined"
        
        spec = network_policy['spec']
        assert 'podSelector' in spec, "Network policy should have pod selector"
        assert 'policyTypes' in spec, "Network policy should have policy types"
        assert 'ingress' in spec, "Network policy should have ingress rules"
        assert 'egress' in spec, "Network policy should have egress rules"

    def test_kubernetes_persistent_volumes(self):
        """Test Kubernetes persistent volume claims"""
        deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        
        with open(deploy_path, 'r') as f:
            deploy_config = yaml.safe_load_all(f)
        
        # Check for PVCs
        pvcs = []
        for resource in deploy_config:
            if resource and resource.get('kind') == 'PersistentVolumeClaim':
                pvcs.append(resource)
        
        assert len(pvcs) > 0, "Should have at least one PVC defined"
        
        required_pvcs = ['medallionx-logs-pvc', 'medallionx-data-pvc', 'medallionx-models-pvc']
        pvc_names = [pvc['metadata']['name'] for pvc in pvcs]
        
        for required_pvc in required_pvcs:
            assert required_pvc in pvc_names, f"PVC {required_pvc} should be defined"
        
        # Check PVC configuration
        for pvc in pvcs:
            spec = pvc['spec']
            assert 'accessModes' in spec, "PVC should have access modes"
            assert 'resources' in spec, "PVC should have resource requests"
            assert 'storageClassName' in spec, "PVC should have storage class"

    def test_kubernetes_ingress_configuration(self):
        """Test Kubernetes ingress configuration"""
        deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        
        with open(deploy_path, 'r') as f:
            deploy_config = yaml.safe_load_all(f)
        
        # Check for Ingress
        ingress = None
        for resource in deploy_config:
            if (resource and resource.get('kind') == 'Ingress' and 
                resource.get('metadata', {}).get('name') == 'medallionx-ingress'):
                ingress = resource
                break
        
        assert ingress is not None, "Ingress should be defined"
        
        spec = ingress['spec']
        assert 'tls' in spec, "Ingress should have TLS configuration"
        assert 'rules' in spec, "Ingress should have rules"
        assert len(spec['rules']) > 0, "Ingress should have at least one rule"
        
        # Check annotations
        metadata = ingress['metadata']
        assert 'annotations' in metadata, "Ingress should have annotations"
        annotations = metadata['annotations']
        assert 'kubernetes.io/ingress.class' in annotations, "Should specify ingress class"
        assert 'nginx.ingress.kubernetes.io/ssl-redirect' in annotations, "Should have SSL redirect"


class TestVaultConfiguration:
    """Test cases for Vault configuration"""
    
    def test_vault_config_structure(self):
        """Test Vault configuration structure"""
        vault_path = Path(__file__).parent.parent / "kubernetes" / "vault.yaml"
        
        with open(vault_path, 'r') as f:
            vault_config = yaml.safe_load_all(f)
        
        # Check for required Vault components
        required_components = [
            'Namespace', 'Deployment', 'Service', 'ServiceAccount',
            'ConfigMap', 'Secret', 'PersistentVolumeClaim', 'Job'
        ]
        found_components = set()
        
        for resource in vault_config:
            if resource and 'kind' in resource:
                found_components.add(resource['kind'])
        
        for component in required_components:
            assert component in found_components, f"Vault component {component} should be defined"

    def test_vault_security_configuration(self):
        """Test Vault security configuration"""
        vault_path = Path(__file__).parent.parent / "kubernetes" / "vault.yaml"
        
        with open(vault_path, 'r') as f:
            vault_config = yaml.safe_load_all(f)
        
        # Find Vault deployment
        vault_deployment = None
        for resource in vault_config:
            if (resource and resource.get('kind') == 'Deployment' and 
                resource.get('metadata', {}).get('name') == 'vault'):
                vault_deployment = resource
                break
        
        assert vault_deployment is not None, "Vault deployment should be defined"
        
        # Check security context
        spec = vault_deployment['spec']
        pod_spec = spec['template']['spec']
        assert 'securityContext' in pod_spec, "Should have pod security context"
        assert pod_spec['securityContext']['runAsUser'] == 100, "Should run as non-root user"
        
        # Check container security
        container = pod_spec['containers'][0]
        assert 'securityContext' in container, "Should have container security context"
        
        # Check TLS configuration
        assert 'volumeMounts' in container, "Should have volume mounts"
        tls_mounts = [mount for mount in container['volumeMounts'] if 'tls' in mount['name']]
        assert len(tls_mounts) > 0, "Should mount TLS certificates"

    def test_vault_auto_unseal_configuration(self):
        """Test Vault auto-unseal configuration"""
        vault_path = Path(__file__).parent.parent / "kubernetes" / "vault.yaml"
        
        with open(vault_path, 'r') as f:
            vault_config = yaml.safe_load_all(f)
        
        # Find Vault config
        vault_config_map = None
        for resource in vault_config:
            if (resource and resource.get('kind') == 'ConfigMap' and 
                resource.get('metadata', {}).get('name') == 'vault-config'):
                vault_config_map = resource
                break
        
        assert vault_config_map is not None, "Vault config map should be defined"
        
        config_data = vault_config_map['data']['vault.hcl']
        
        # Check for auto-unseal configuration
        assert 'seal "awskms"' in config_data, "Should have AWS KMS seal configuration"
        assert 'storage "raft"' in config_data, "Should have Raft storage backend"
        assert 'ui = true' in config_data, "Should have UI enabled"
        assert 'listener "tcp"' in config_data, "Should have TCP listener"

    def test_vault_init_job(self):
        """Test Vault initialization job"""
        vault_path = Path(__file__).parent.parent / "kubernetes" / "vault.yaml"
        
        with open(vault_path, 'r') as f:
            vault_config = yaml.safe_load_all(f)
        
        # Find init job
        init_job = None
        for resource in vault_config:
            if (resource and resource.get('kind') == 'Job' and 
                resource.get('metadata', {}).get('name') == 'vault-init'):
                init_job = resource
                break
        
        assert init_job is not None, "Vault init job should be defined"
        
        # Check job configuration
        spec = init_job['spec']
        assert spec['restartPolicy'] == 'OnFailure', "Should restart on failure"
        
        pod_spec = spec['template']['spec']
        container = pod_spec['containers'][0]
        
        # Check init script content
        command = container['command']
        assert len(command) > 1, "Should have initialization script"
        
        # Check for Vault operations
        init_script = command[-1]
        assert 'vault operator init' in init_script, "Should initialize Vault"
        assert 'vault operator unseal' in init_script, "Should unseal Vault"
        assert 'vault auth enable' in init_script, "Should enable auth methods"

    def test_vault_external_secrets(self):
        """Test Vault external secrets configuration"""
        vault_path = Path(__file__).parent.parent / "kubernetes" / "vault.yaml"
        
        with open(vault_path, 'r') as f:
            vault_config = yaml.safe_load_all(f)
        
        # Check for ExternalSecret resources
        external_secrets = []
        for resource in vault_config:
            if resource and resource.get('kind') == 'ExternalSecret':
                external_secrets.append(resource)
        
        assert len(external_secrets) > 0, "Should have external secrets defined"
        
        # Check secret store
        secret_store = None
        for resource in vault_config:
            if (resource and resource.get('kind') == 'SecretStore' and 
                resource.get('metadata', {}).get('name') == 'vault-store'):
                secret_store = resource
                break
        
        assert secret_store is not None, "Should have secret store defined"
        
        # Check secret store configuration
        spec = secret_store['spec']
        assert 'provider' in spec, "Should have provider configuration"
        assert 'vault' in spec['provider'], "Should have Vault provider"
        assert 'server' in spec['provider']['vault'], "Should specify Vault server"


class TestInfrastructureIntegration:
    """Integration tests for infrastructure components"""
    
    def test_docker_to_kubernetes_consistency(self):
        """Test consistency between Docker and Kubernetes configurations"""
        # Check that services defined in Docker Compose have corresponding K8s deployments
        docker_compose_path = Path(__file__).parent.parent / "docker" / "deployment" / "docker-compose.yml"
        k8s_deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        
        with open(docker_compose_path, 'r') as f:
            docker_config = yaml.safe_load(f)
        
        with open(k8s_deploy_path, 'r') as f:
            k8s_config = yaml.safe_load_all(f)
        
        # Get Docker services
        docker_services = set(docker_config['services'].keys())
        
        # Get K8s deployments
        k8s_deployments = set()
        for resource in k8s_config:
            if resource and resource.get('kind') == 'Deployment':
                name = resource['metadata']['name']
                # Extract base name (remove suffixes like -deployment)
                base_name = name.replace('-deployment', '').replace('-engine', '')
                k8s_deployments.add(base_name)
        
        # Check for key services
        key_services = {'trading-engine', 'dashboard', 'postgres', 'redis'}
        for service in key_services:
            assert service in docker_services, f"Service {service} should be in Docker Compose"
            # Note: Not all Docker services need K8s equivalents (e.g., nginx)

    def test_environment_variable_consistency(self):
        """Test environment variable consistency across configurations"""
        # Check that critical environment variables are defined consistently
        docker_compose_path = Path(__file__).parent.parent / "docker" / "deployment" / "docker-compose.yml"
        k8s_deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        
        with open(docker_compose_path, 'r') as f:
            docker_config = yaml.safe_load(f)
        
        with open(k8s_deploy_path, 'r') as f:
            k8s_config = yaml.safe_load_all(f)
        
        # Get K8s environment variables
        k8s_env_vars = set()
        for resource in k8s_config:
            if resource and resource.get('kind') == 'Deployment':
                containers = resource['spec']['template']['spec']['containers']
                for container in containers:
                    if 'env' in container:
                        for env_var in container['env']:
                            k8s_env_vars.add(env_var['name'])
        
        # Check for critical environment variables
        critical_env_vars = {
            'ENVIRONMENT', 'REDIS_URL', 'POSTGRES_URL', 
            'BINANCE_API_KEY', 'BINANCE_SECRET'
        }
        
        for env_var in critical_env_vars:
            # Should be defined in at least one configuration
            docker_defined = any(
                env_var in str(service_config.get('environment', {}))
                for service_config in docker_config['services'].values()
            )
            k8s_defined = env_var in k8s_env_vars
            
            # At least one should be defined
            assert docker_defined or k8s_defined, f"Environment variable {env_var} should be defined"

    def test_resource_limits_consistency(self):
        """Test resource limits consistency across configurations"""
        docker_compose_path = Path(__file__).parent.parent / "docker" / "deployment" / "docker-compose.yml"
        k8s_deploy_path = Path(__file__).parent.parent / "kubernetes" / "deployment.yaml"
        
        with open(docker_compose_path, 'r') as f:
            docker_config = yaml.safe_load(f)
        
        with open(k8s_deploy_path, 'r') as f:
            k8s_config = yaml.safe_load_all(f)
        
        # Check that critical services have resource limits in both configurations
        critical_services = ['trading-engine', 'dashboard']
        
        for service in critical_services:
            # Check Docker Compose
            if service in docker_config['services']:
                docker_service = docker_config['services'][service]
                if 'deploy' in docker_service and 'resources' in docker_service['deploy']:
                    assert 'limits' in docker_service['deploy']['resources'], f"Docker service {service} should have resource limits"
            
            # Check Kubernetes
            k8s_deployment_name = f"medallionx-{service}" if service != 'trading-engine' else "medallionx-trading-engine"
            for resource in k8s_config:
                if (resource and resource.get('kind') == 'Deployment' and 
                    resource.get('metadata', {}).get('name') == k8s_deployment_name):
                    containers = resource['spec']['template']['spec']['containers']
                    for container in containers:
                        assert 'resources' in container, f"K8s deployment {k8s_deployment_name} should have resource limits"
                        assert 'limits' in container['resources'], f"K8s deployment {k8s_deployment_name} should have resource limits"
                    break


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
