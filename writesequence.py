import os

def store_labels_as_fa(labels, folder):
    

    os.makedirs(folder, exist_ok=True)
    for i, sequence in enumerate(labels):
        trimmed_sequence = sequence 
        if sequence[0] == 'E' or sequence[0] == 'B':
            trimmed_sequence = sequence[1:]

        if sequence[-1] == 'E' or sequence[-1] == 'B':
            trimmed_sequence = trimmed_sequence[:-1]

        trimmed_sequence_str = ''.join(trimmed_sequence)  
        reversed_sequence = trimmed_sequence_str[::-1]  
        
        graph_prefix = os.path.basename(os.path.dirname(folder))
        file_path = os.path.join(folder, 'output_{}_{}.fa'.format(graph_prefix, i))
        with open(file_path, 'w') as fa_file:
            fa_file.write('>{}_sequence{}    num:{} \n'.format(folder, i, len(trimmed_sequence)))
            fa_file.write(reversed_sequence + '\n')
    
    return file_path



def store_labels_as_fa_hanshuming(labels, folder, filename_prefix):
   
    os.makedirs(folder, exist_ok=True)

    for i, sequence in enumerate(labels):
        trimmed_sequence = sequence  

       
        if sequence[0] == 'E' or sequence[0] == 'B':
            trimmed_sequence = sequence[1:]
        if sequence[-1] == 'E' or sequence[-1] == 'B':
            trimmed_sequence = trimmed_sequence[:-1]

        trimmed_sequence_str = ''.join(trimmed_sequence) 
        reversed_sequence = trimmed_sequence_str[::-1]  

        
        file_path = os.path.join(folder, f'{filename_prefix}_{i}.fa')

        with open(file_path, 'w') as fa_file:
            fa_file.write(f'>{filename_prefix}_sequence{i}    num:{len(trimmed_sequence)}\n')
            fa_file.write(reversed_sequence + '\n')

    return file_path
