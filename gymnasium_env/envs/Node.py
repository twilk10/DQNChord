
class Node:
    def __init__(self, id, active_status):
        self.is_active = active_status
        self.id = id
        self.is_agent = True if self.id == 0 else False
        self.finger_table = {
            'predecessors': [],
            'successors': []
        }
        self.data={}
        
    def set_active_status(self, new_status):
        self.is_active = new_status
    def __str__(self):
        return (f"\t Node Id: {self.id}\n"
                f"\t Active Status: {self.is_active}\n"
                f"\t Finger Table: {self.finger_table}\n")
