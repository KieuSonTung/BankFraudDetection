from collections import Counter

def hard_voting(matrix):
    # Transpose the matrix to group values by position
    transposed_matrix = list(map(list, zip(*matrix)))
    
    # Initialize a list to store the results
    result = []
    
    # Iterate through the transposed matrix
    for elements in transposed_matrix:
        # Count the occurrences of each element in the current position
        vote_counts = Counter(elements)
        
        # Find the most common element (mode)
        most_common_value, _ = vote_counts.most_common(1)[0]
        
        # Append the mode to the result list
        result.append(most_common_value)
    
    return result

def soft_voting(matrix):
    
    return [sum(col) / len(col) for col in zip(*matrix)]