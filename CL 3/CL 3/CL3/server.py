import random
import itertools

class Server:
    def __init__(self, server_id, weight=1):
        self.server_id = server_id
        self.active_connections = 0
        self.total_requests = 0
        self.busy_time = 0
        self.weight = weight

    def handle_request(self, processing_time):
        self.active_connections += 1
        self.total_requests += 1
        self.busy_time += processing_time

    def release_request(self):
        if self.active_connections > 0:
            self.active_connections -= 1

    def utilization(self, total_time):
        return (self.busy_time / total_time) * 100 if total_time > 0 else 0

    def __repr__(self):
        return f"Server-{self.server_id} (Active: {self.active_connections}, Total Requests: {self.total_requests})"

class LoadBalancer:
    def __init__(self, servers, algorithm="round_robin"):
        self.servers = servers
        self.algorithm = algorithm
        self.rr_iterator = itertools.cycle(self.servers)
        self.wrr_list = self._build_weighted_list()
        self.wrr_iterator = itertools.cycle(self.wrr_list)
        self.total_time = 0
        self.requests_handled = 0
        self.wait_times = []

    def _build_weighted_list(self):
        weighted = []
        for server in self.servers:
            weighted.extend([server] * server.weight)
        return weighted

    def get_server(self):
        if self.algorithm == "round_robin":
            return next(self.rr_iterator)
        elif self.algorithm == "least_connections":
            return min(self.servers, key=lambda s: s.active_connections)
        elif self.algorithm == "weighted_round_robin":
            return next(self.wrr_iterator)
        else:
            raise ValueError("Unsupported algorithm")

    def distribute_request(self, request_id, arrival_time, processing_time):
        server = self.get_server()
        server.handle_request(processing_time)
        self.requests_handled += 1
        self.total_time = max(self.total_time, arrival_time + processing_time)
        self.wait_times.append(arrival_time)
        print(f"Request-{request_id} → {server} | Processing Time: {processing_time}")

    def print_summary(self):
        print("\n--- Simulation Summary ---")
        print(f"Total Requests Processed: {self.requests_handled}")
        avg_wait = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        print(f"Average Waiting Time: {avg_wait:.2f} time units")
        for server in self.servers:
            print(f"{server}: Utilization = {server.utilization(self.total_time):.2f}%")
