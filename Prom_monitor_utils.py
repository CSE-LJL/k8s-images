import requests


def fetch_cpu_usage():
    url = 'http://172.18.233.33:30090/api/v1/query'
    # CPU
    query = "(1-((sum(increase(node_cpu_seconds_total{mode='idle'}[1m])) by (instance))/(sum(increase(node_cpu_seconds_total[1m])) by (instance))))"
    # Mem
    # query = "((node_memory_MemTotal_bytes - node_memory_MemFree_bytes - node_memory_Buffers_bytes - node_memory_Cached_bytes) / (node_memory_MemTotal_bytes )) * 100"
    params = {
        'query': query
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch CPU usage data from Prometheus. Status code: {response.status_code}")
        return None


def fetch_gpu_usage():
    url = 'http://172.18.233.33:30090/api/v1/query'
    #
    query1 = "DCGM_FI_DEV_GPU_UTIL"
    query2 = "DCGM_FI_DEV_MEM_COPY_UTIL"
    #
    # query = "((node_memory_MemTotal_bytes - node_memory_MemFree_bytes - node_memory_Buffers_bytes - node_memory_Cached_bytes) / (node_memory_MemTotal_bytes )) * 100"
    params = {
        'query': query1
    }
    response1 = requests.get(url, params=params)

    params = {
        'query': query2
    }
    response2 = requests.get(url, params=params)

    if response1.status_code == 200 and response2.status_code == 200:
        data1 = response1.json()
        data2 = response2.json()
        return {'gpu_util': data1, 'gpu_mem': data2}
    else:
        print(f"Failed to fetch GPU usage data from Prometheus. Status code: {response1.status_code}")
        return None


def fetch_mem_usage():
    url = 'http://172.18.233.33:30090/api/v1/query'
    # CPU
    # query = "(1-((sum(increase(node_cpu_seconds_total{mode='idle'}[1m])) by (instance))/(sum(increase(node_cpu_seconds_total[1m])) by (instance)))) * 100"
    # Mem
    query = "((node_memory_MemTotal_bytes - node_memory_MemFree_bytes - node_memory_Buffers_bytes - node_memory_Cached_bytes) / (node_memory_MemTotal_bytes ))"
    params = {
        'query': query
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch Mem usage data from Prometheus. Status code: {response.status_code}")
        return None


GPU_hostnames = {'dcgm-exporter-6c68d': 'ubuntu-2080ti', 'dcgm-exporter-8sltv': 'lijlun3-ndc-server-3090-2',
                 'dcgm-exporter-d7wzq': 'lijlun3-1080ti-server', 'dcgm-exporter-gqfm9': 'lijlun3-ndc-3090',
                 'dcgm-exporter-mq6dv': 'ubuntu-01', 'dcgm-exporter-p6g8v': 'ubuntu-03',
                 'dcgm-exporter-prlxl': 'ubuntu-02'}

CPU_hostnames = {'172.18.167.22': 'ubuntu-2080ti', '172.18.167.23': 'ubuntu-03', '172.18.167.25': 'ubuntu-02',
                '172.18.233.41': 'ubuntu-01', '172.18.232.123': 'lijlun3-1080Ti-Server', '172.18.232.124': 'lijlun3-ndc-3090',
                '172.18.233.39': 'lijlun3-ndc-server-3090-2'}


def cpu_data_parser(raw_data):
    cpu_info = {}
    cpu_data = raw_data['data']['result']
    for item in cpu_data:
        cpu_info[CPU_hostnames[item['metric']['instance'].split(':')[0]]] = item['value'][1]
    return cpu_info


def mem_data_parser(raw_data):
    mem_info = {}
    mem_data = raw_data['data']['result']
    for item in mem_data:
        mem_info[CPU_hostnames[item['metric']['instance'].split(':')[0]]] = item['value'][1]
    return mem_info


def gpu_data_parser(raw_data):
    gpu_util_info = {}
    gpu_mem_info = {}
    gpu_util_data = raw_data['gpu_util']['data']['result']
    gpu_mem_data = raw_data['gpu_mem']['data']['result']
    for item in gpu_util_data:
        if GPU_hostnames[item['metric']['Hostname']] not in gpu_util_info.keys():
            gpu_util_info[GPU_hostnames[item['metric']['Hostname']]] = {'device_type': item['metric']['modelName'], 'usage': []}
        gpu_util_info[GPU_hostnames[item['metric']['Hostname']]]['usage'].append(item['value'][1])
    for item in gpu_mem_data:
        if GPU_hostnames[item['metric']['Hostname']] not in gpu_mem_info.keys():
            gpu_mem_info[GPU_hostnames[item['metric']['Hostname']]] = {'device_type': item['metric']['modelName'], 'usage': []}
        gpu_mem_info[GPU_hostnames[item['metric']['Hostname']]]['usage'].append(item['value'][1])
    return {'gpu_util': gpu_util_info, 'gpu_mem': gpu_mem_info}


if __name__ == '__main__':
    cpu_result = fetch_cpu_usage()
    # print(type(cpu_result))
    mem_result = fetch_mem_usage()
    gpu_result = fetch_gpu_usage()
    if cpu_result:
        print(f"CPU usage data: {cpu_data_parser(cpu_result)}")
        # 在这里可以进一步处理返回的数据，例如解析和展示
    else:
        print("Failed to fetch CPU usage data.")
    if mem_result:
        print(f"Mem usage data: {mem_data_parser(mem_result)}")
        # 在这里可以进一步处理返回的数据，例如解析和展示
    else:
        print("Failed to fetch Mem usage data.")
    if gpu_result:
        print(f"GPU usage data: {gpu_data_parser(gpu_result)}")
        # 在这里可以进一步处理返回的数据，例如解析和展示
    else:
        print("Failed to fetch GPU usage data.")
