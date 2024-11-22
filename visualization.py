import time
from pyvis.network import Network

def visualize_chord_network(chord_network):
    net = Network(directed=True, notebook=False)
    net.toggle_physics(True)

    nodes, edges = chord_network.get_visualization_data()
    # print('node list size', nodes)

    # Add nodes to the network
    for node in nodes:
        net.add_node(n_id=node['id'], label=node['label'], title=node['title'], 
                     color=node['color'], x=node['x'], y=node['y'])

    # Add edges to the network
    # print('[edges]', edges)
    for edge in edges:
        net.add_edge(source=edge['from'], to=edge['to'],  arrows=edge.get('arrows', 'to'), 
                     color='blue', width=2)

    # Generate and show the network graph
    net.show('chord_network.html', notebook= False)

def run_visualization(chord_network, interval=5):
    while True:
        visualize_chord_network(chord_network)
        time.sleep(interval)