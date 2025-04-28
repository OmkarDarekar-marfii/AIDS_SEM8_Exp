import random
from server import Server, LoadBalancer

def generate_request(request_id):
    arrival_time = request_id
    processing_time = random.randint(1, 5)
    return request_id, arrival_time, processing_time

def simulate_requests(num_requests=15, num_servers=3, algorithm="round_robin"):
    servers = [Server(i+1, weight=random.randint(1, 3)) for i in range(num_servers)]
    lb = LoadBalancer(servers, algorithm)

    for i in range(num_requests):
        req_id, arrival, duration = generate_request(i+1)
        lb.distribute_request(req_id, arrival, duration)

    lb.print_summary()

if __name__ == "__main__":
    simulate_requests(num_requests=15, num_servers=3, algorithm="least_connections")
