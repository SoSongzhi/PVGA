sequence = "example"

if sequence[0] == 'e' or sequence[0] == 'b':
    trimmed_sequence = sequence[1:]

if sequence[-1] == 'e' or sequence[-1] == 'b':
    trimmed_sequence = trimmed_sequence[:-1]
print(trimmed_sequence)