import threading

class Repeated(threading.Thread):

    def __init__(self, vehicle, stop_event):
        threading.Thread.__init__(self)
        self.stop_event = stop_event
        self.damon = True
        self.vehicle = vehicle

    def run(self):
        while True:
            if not self.stop_event.is_set():
                self.vehicle.drive_straight(0.2)
            else:
                self.stop()