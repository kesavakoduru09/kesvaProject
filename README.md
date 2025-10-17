# Breadth-First Search (BFS) Program in Python

# Graph representation using a dictionary
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

# List for visited nodes
visited = []

# Queue for BFS
queue = []

# Function for BFS traversal
def bfs(visited, graph, node):
    visited.append(node)
    queue.append(node)

    while queue:
        # Remove the first node from the queue
        m = queue.pop(0)
        print(m, end=" ")

        # Visit all the adjacent nodes
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

# Driver Code
print("Following is the Breadth-First Search:")
bfs(visited, graph, '5')
