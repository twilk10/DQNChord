from Chord import ChordNetwork
from visualization import run_visualization
import threading
import time

# Initialize the Chord network
my_chord_network = ChordNetwork(size= 10, r = 2, bank_size=20, start_timer_thread= False)
# run_visualization(my_chord_network)

# Start the network's timer thread if it's not started within the ChordNetwork class
# chord_network.start_timer_thread()

# Start the visualization thread
visualization_thread = threading.Thread(target=run_visualization, args=(my_chord_network,))
visualization_thread.daemon = True
visualization_thread.start()

# If using Flask for real-time updates
# from app import run_flask_app
# flask_thread = threading.Thread(target=run_flask_app)
# flask_thread.daemon = True
# flask_thread.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down.")
