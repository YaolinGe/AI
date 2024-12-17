import matplotlib.pyplot as plt

def birthday_paradox(num_people):
    probability = 1.0
    for i in range(num_people):
        probability *= (365 - i) / 365
    return probability

def plot_birthday_paradox(max_people):
    people = list(range(1, max_people + 1))
    probabilities = [birthday_paradox(n) for n in people]

    plt.figure(figsize=(10, 6))
    plt.plot(people, probabilities, marker='o')
    plt.title('Probability of No Shared Birthdays')
    plt.xlabel('Number of People')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    max_people = 100
    plot_birthday_paradox(max_people)