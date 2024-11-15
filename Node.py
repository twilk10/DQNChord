
class Node:
    def __init__(self, id , ttl, active_status):
        self.max_time = ttl
        self.ttl = ttl
        self.is_active: bool = active_status
        self.id = id
        self.finger_table = {
            'predecessors': [],
            'successors':[]
        }
   
    def set_active_status(self, new_status):
        self.is_active = new_status

    def reset_timer(self):
        self.ttl = self.max_time

    def __str__(self):
        return f"\t Node Id:{self.id} \n \t Active Status:{self.is_active}  \n \t ttl:{self.ttl} \n \t Finger Table: {self.finger_table} \n"
   




