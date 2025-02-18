from decimal import *
getcontext().prec = 100

# Estimates hidden state path given parameters and inputs
def ForwardBackward(states, input, transitions, emissions):
    forward = []
    backward = []
    probs = []
    probs2 = []

    # Initial forward and backward arrays
    for k in range(len(states)):
        forward.append([Decimal(0)] * (len(input)))
        backward.append([Decimal(0)] * (len(input)))
        probs.append([Decimal(0)] * (len(input)))
        probs2.append([])
        for j in range(len(states)):
            probs2[k].append([Decimal(0)] * (len(input) - 1))
        forward[k][0] = Decimal(1 / len(states)) * emissions[states[k]][input[0]]
        backward[k][-1] = Decimal(1)

    # Compute forwards
    for i in range(1, len(input)):
        for k in range(len(states)):
            for l in range(len(states)):
                prob = forward[l][i - 1] * transitions[states[l]][states[k]] * emissions[states[k]][input[i]]
                forward[k][i] += prob
        
    # Compute backwards
    for i in reversed(range(len(input) - 1)):
        for k in range(len(states)):
            for l in range(len(states)):
                prob = backward[l][i + 1] * transitions[states[k]][states[l]] * emissions[states[l]][input[i + 1]]
                backward[k][i] += prob

    sink = Decimal(0)
    for k in range(len(states)):
        sink += forward[k][-1]

    # Compute probabilities
    for i in range(len(input)):
        for k in range(len(states)):
            prob = forward[k][i] * backward[k][i] / sink
            probs[k][i] = prob

    for i in range(len(input) - 1):
        for l in range(len(states)):
            for k in range(len(states)):
                probs2[l][k][i] = forward[l][i] * transitions[states[l]][states[k]] * emissions[states[k]][input[i+1]] * backward[k][i + 1] / sink

    return probs, probs2

# Expectation maximization for parameter estimation
def BaumWelch(states, alphabet, symbols, initial_transitions, initial_emissions, j):
    # Transition and emission probability matrices
    transitions = initial_transitions
    emissions = initial_emissions

    # Iterate for a set number of iterations to produce better results
    for iteration in range(j):
        # Expectation step
        probs, probs2 = ForwardBackward(states, symbols, transitions, emissions)

        # Maximization step
        transitions = {}
        emissions = {}
        for l in states:
            transitions[l] = {}
            emissions[l] = {}
            for k in states:
                transitions[l][k] = 0
            for a in alphabet:
                emissions[l][a] = 0

        for i in range(len(symbols) - 1):
            for l in range(len(states)):
                for k in range(len(states)):
                    transitions[states[l]][states[k]] += probs2[l][k][i]

        for i in range(len(symbols)):
            for l in range(len(states)):
                emissions[states[l]][symbols[i]] += probs[l][i]
        

        # Normalization
        for state in transitions:
            total = 0
            for key in transitions[state]:
                total += transitions[state][key]
            for key in transitions[state]:
                if total != 0:
                    transitions[state][key] /= total
                else:
                    transitions[state][key] = 1 / len(states)
        

        for state in emissions:
            total = 0
            for key in emissions[state]:
                total += emissions[state][key]
            for key in emissions[state]:
                emissions[state][key] /= total

    final_probs, final_probs2 =  ForwardBackward(states, symbols, transitions, emissions)
    seq_to_score = enumerate(final_probs[1])
    seq_to_score = sorted(seq_to_score, key=lambda x: x[1])

    
    # Write predictions to a file
    f = open("predictions.txt", "w")

    for i in range(1, 50001):
        f.write(str(seq_to_score[-i][0]) + "\n")

    f.close()

    return transitions, emissions


txt = ""
count = 0
with open("input.fasta") as file:
    for line in file:
        if count != 0:
            txt += line[:-1]
        count += 1

#print(txt[:100])

# Transition matrix with initial guesses
t = {
    "A": {
        "A": Decimal(0.55),
        "B": Decimal(0.45)
    },
    "B": {
        "A": Decimal(0.45),
        "B": Decimal(0.55)
    }
}

# Emissions matrix with initial guesses
e = {
    "A": {
        "n": Decimal(0.75),
        "x": Decimal(0.1),
        "y": Decimal(0.1),
        "z": Decimal(0.05)
    },
    "B": {
        "n": Decimal(0.1),
        "x": Decimal(0.2),
        "y": Decimal(0.2),
        "z": Decimal(0.5)
    }
}

a = ["n", "x", "y", "z"]
s = ["A", "B"]


# Use (20 iterations of) Baum-Welch algorithm to find parameters of Hidden Markov Model 
transitions, emissions = BaumWelch(s, a, txt, t, e, 5)

for state in transitions:
    txt = state + "\t"
    for st in s:
        txt += str(transitions[state][st]) + "\t"
    print(txt)
    
print()

for state in emissions:
    txt = state + "\t"
    for key in a:
        txt += str(emissions[state][key]) + "\t"
    print(txt)


                    

        



