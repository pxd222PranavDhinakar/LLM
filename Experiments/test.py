import random

def generate_test_problems(length1, length2):
    """Generate test addition problems of specified lengths using string manipulation"""
    def generate_n_digit_number(n):
        # First digit can't be 0
        first_digit = random.randint(1, 9)
        # Rest of digits can be 0-9
        rest_digits = [random.randint(0, 9) for _ in range(n-1)]
        # Combine digits
        num = int(str(first_digit) + ''.join(map(str, rest_digits)))
        return num

    num1 = generate_n_digit_number(length1)
    num2 = generate_n_digit_number(length2)
    result = num1 + num2
    return num1, num2, result

# Set random seed for reproducibility
random.seed(42)

# Test a few different length combinations
test_lengths = [(1,1), (2,2), (3,3), (5,3), (10,8), (15,12), (20,20)]

print("Sample test problems:")
print("-" * 50)
for len1, len2 in test_lengths:
    num1, num2, result = generate_test_problems(len1, len2)
    print(f"\nLength {len1} + Length {len2}:")
    print(f"{num1} + {num2} = {result}")
    print(f"Lengths: {len(str(num1))}, {len(str(num2))}, {len(str(result))}")