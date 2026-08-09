import time


class Info:
    def __init__(self):
        self.time = 0
        self.errors = []
        self.status = "In Progress"
        self.objects = []

    def get_info(self):
        output = [
            f"Status: {self.status}",
            f"Time: {self.time:.2f}s",
            f"Objects: {', '.join(self.objects)}",
        ]
        if self.errors:
            output.append("Errors:")
            output.extend(self.errors)
        return output


class Logger:
    def __init__(self):
        self.unwrap_info = []
        self.start_time = 0

    def new_info(self):
        self.unwrap_info.append(Info())
        self.start_timer()

    def discard_info(self):
        """Drop the entry for a run that was refused before it started."""
        self.unwrap_info.pop()

    def add_data(self, target, data):
        getattr(self.get_latest(), target).append(data)

    def change_status(self, status):
        self.get_latest().status = status

    def get_latest(self):
        # if logs cleared during unwrap, add a new one
        if not self.unwrap_info:
            self.new_info()
        return self.unwrap_info[-1]

    def get_all(self):
        """Every run's info, newest first, blank separated."""
        output = []
        for info in reversed(self.unwrap_info):
            output.extend(info.get_info())
            output.append("")
        return output[:-1]

    def start_timer(self):
        self.start_time = time.perf_counter()

    def update_time(self):
        self.get_latest().time = time.perf_counter() - self.start_time


logger = Logger()
