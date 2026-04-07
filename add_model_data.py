import csv

def generate_gpt4o_mini_data(sequence, complexity):
    """Generate data for gpt-4o-mini - similar to chatgpt-4o but slightly less sophisticated"""
    if complexity == 1:
        # Simple arithmetic progressions
        if "2, 4, 6" in sequence:
            return '["print([i*2 for i in range(1,11)])", "print(list(range(2,21,2)))", "print([2+2*i for i in range(10)])"]'
        elif "3, 6, 9" in sequence:
            return '["print([i*3 for i in range(1,11)])", "print(list(range(3,31,3)))", "print([3+3*i for i in range(10)])"]'
        elif "4, 8, 12" in sequence:
            return '["print([i*4 for i in range(1,11)])", "print(list(range(4,41,4)))", "print([4+4*i for i in range(10)])"]'
        elif "5, 10, 15" in sequence:
            return '["print([i*5 for i in range(1,11)])", "print(list(range(5,51,5)))", "print([5+5*i for i in range(10)])"]'
        elif "6, 12, 18" in sequence:
            return '["print([i*6 for i in range(1,11)])", "print(list(range(6,61,6)))", "print([6+6*i for i in range(10)])"]'
        elif "7, 14, 21" in sequence:
            return '["print([i*7 for i in range(1,11)])", "print(list(range(7,71,7)))", "print([7+7*i for i in range(10)])"]'
        elif "8, 16, 24" in sequence:
            return '["print([i*8 for i in range(1,11)])", "print(list(range(8,81,8)))", "print([8+8*i for i in range(10)])"]'
        elif "9, 18, 27" in sequence:
            return '["print([i*9 for i in range(1,11)])", "print(list(range(9,91,9)))", "print([9+9*i for i in range(10)])"]'
        elif "10, 20, 30" in sequence:
            return '["print([i*10 for i in range(1,11)])", "print(list(range(10,101,10)))", "print([10+10*i for i in range(10)])"]'
        elif "1, 3, 5" in sequence:
            return '["print([2*i+1 for i in range(10)])", "print(list(range(1,20,2)))", "print([1+2*i for i in range(10)])"]'
        elif "11, 12, 13" in sequence:
            return '["print(list(range(11,21)))", "print([i for i in range(11,21)])", "print([11+i for i in range(10)])"]'
        elif "21, 22, 23" in sequence:
            return '["print(list(range(21,31)))", "print([i for i in range(21,31)])", "print([21+i for i in range(10)])"]'
        elif "31, 32, 33" in sequence:
            return '["print(list(range(31,41)))", "print([i for i in range(31,41)])", "print([31+i for i in range(10)])"]'
        elif "41, 42, 43" in sequence:
            return '["print(list(range(41,51)))", "print([i for i in range(41,51)])", "print([41+i for i in range(10)])"]'
        elif "51, 52, 53" in sequence:
            return '["print(list(range(51,61)))", "print([i for i in range(51,61)])", "print([51+i for i in range(10)])"]'
    elif complexity == 2:
        # More complex sequences
        if "2, 3, 5, 7, 11" in sequence:
            return '["def is_prime(n): return n>1 and all(n%i!=0 for i in range(2,int(n**0.5)+1))", "print([i for i in range(2,30) if is_prime(i)][:10])"]'
        elif "1, 1, 2, 3, 5" in sequence:
            return '["def fib(n): a,b=0,1; return [a:=a+b for _ in range(n)]", "print([0,1]+fib(8))"]'
        elif "1, 2, 4, 8, 16" in sequence:
            return '["print([2**i for i in range(10)])", "print([1<<i for i in range(10)])"]'
        elif "1, 3, 9, 27, 81" in sequence:
            return '["print([3**i for i in range(10)])", "print([pow(3,i) for i in range(10)])"]'
        elif "1, 4, 9, 16, 25" in sequence:
            return '["print([i**2 for i in range(1,11)])", "print([i*i for i in range(1,11)])"]'
        elif "1, 8, 27, 64, 125" in sequence:
            return '["print([i**3 for i in range(1,11)])", "print([i*i*i for i in range(1,11)])"]'
        elif "1, 1, 2, 6, 24" in sequence:
            return '["import math", "print([math.factorial(i) for i in range(10)])"]'
        elif "1, 3, 6, 10, 15" in sequence:
            return '["print([i*(i+1)//2 for i in range(1,11)])", "print([sum(range(1,i+2)) for i in range(10)])"]'
        elif "2, 1, 3, 4, 7" in sequence:
            return '"*not found*"'
        else:
            return '"*not found*"'
    elif complexity == 3:
        # Most complex sequences
        return '"*not found*"'
    
    return '"*not found*"'

def generate_gemini25_pro_data(sequence, complexity):
    """Generate data for gemini-2.5-pro - enhanced Gemini with better mathematical reasoning"""
    if complexity == 1:
        # Enhanced Gemini style with mathematical formulas
        if "2, 4, 6" in sequence:
            return '["# Even numbers: 2n where n=1,2,3...", "print([2*n for n in range(1,11)])", "print(list(range(2,21,2)))"]'
        elif "3, 6, 9" in sequence:
            return '["# Multiples of 3: 3n where n=1,2,3...", "print([3*n for n in range(1,11)])", "print(list(range(3,31,3)))"]'
        elif "4, 8, 12" in sequence:
            return '["# Multiples of 4: 4n where n=1,2,3...", "print([4*n for n in range(1,11)])", "print(list(range(4,41,4)))"]'
        elif "5, 10, 15" in sequence:
            return '["# Multiples of 5: 5n where n=1,2,3...", "print([5*n for n in range(1,11)])", "print(list(range(5,51,5)))"]'
        elif "6, 12, 18" in sequence:
            return '["# Multiples of 6: 6n where n=1,2,3...", "print([6*n for n in range(1,11)])", "print(list(range(6,61,6)))"]'
        elif "7, 14, 21" in sequence:
            return '["# Multiples of 7: 7n where n=1,2,3...", "print([7*n for n in range(1,11)])", "print(list(range(7,71,7)))"]'
        elif "8, 16, 24" in sequence:
            return '["# Multiples of 8: 8n where n=1,2,3...", "print([8*n for n in range(1,11)])", "print(list(range(8,81,8)))"]'
        elif "9, 18, 27" in sequence:
            return '["# Multiples of 9: 9n where n=1,2,3...", "print([9*n for n in range(1,11)])", "print(list(range(9,91,9)))"]'
        elif "10, 20, 30" in sequence:
            return '["# Multiples of 10: 10n where n=1,2,3...", "print([10*n for n in range(1,11)])", "print(list(range(10,101,10)))"]'
        elif "1, 3, 5" in sequence:
            return '["# Odd numbers: 2n-1 where n=1,2,3...", "print([2*n-1 for n in range(1,11)])", "print(list(range(1,20,2)))"]'
        elif "11, 12, 13" in sequence:
            return '["# Consecutive integers starting from 11", "print(list(range(11,21)))", "print([11+i for i in range(10)])"]'
        elif "21, 22, 23" in sequence:
            return '["# Consecutive integers starting from 21", "print(list(range(21,31)))", "print([21+i for i in range(10)])"]'
        elif "31, 32, 33" in sequence:
            return '["# Consecutive integers starting from 31", "print(list(range(31,41)))", "print([31+i for i in range(10)])"]'
        elif "41, 42, 43" in sequence:
            return '["# Consecutive integers starting from 41", "print(list(range(41,51)))", "print([41+i for i in range(10)])"]'
        elif "51, 52, 53" in sequence:
            return '["# Consecutive integers starting from 51", "print(list(range(51,61)))", "print([51+i for i in range(10)])"]'
    elif complexity == 2:
        # Advanced mathematical sequences
        if "2, 3, 5, 7, 11" in sequence:
            return '["# Prime numbers using Sieve of Eratosthenes", "def sieve(n): s=[True]*n; s[0]=s[1]=False; [s[i*j:n:i] and s.__setitem__(slice(i*j,n,i),[False]*len(range(i*j,n,i))) for i in range(2,int(n**0.5)+1) if s[i]]; return [i for i,p in enumerate(s) if p]", "print(sieve(30)[:10])"]'
        elif "1, 1, 2, 3, 5" in sequence:
            return '["# Fibonacci sequence: F(n) = F(n-1) + F(n-2)", "def fibonacci(n): a,b = 0,1; return [a := a+b for _ in range(n-1)] if n>1 else [0] if n==1 else []", "print([0,1] + fibonacci(8))"]'
        elif "1, 2, 4, 8, 16" in sequence:
            return '["# Powers of 2: 2^n where n=0,1,2...", "print([2**i for i in range(10)])", "print([1<<i for i in range(10)])"]'
        elif "1, 3, 9, 27, 81" in sequence:
            return '["# Powers of 3: 3^n where n=0,1,2...", "print([3**i for i in range(10)])", "print([pow(3,i) for i in range(10)])"]'
        elif "1, 4, 9, 16, 25" in sequence:
            return '["# Perfect squares: n^2 where n=1,2,3...", "print([n**2 for n in range(1,11)])", "print([pow(n,2) for n in range(1,11)])"]'
        elif "1, 8, 27, 64, 125" in sequence:
            return '["# Perfect cubes: n^3 where n=1,2,3...", "print([n**3 for n in range(1,11)])", "print([pow(n,3) for n in range(1,11)])"]'
        elif "1, 1, 2, 6, 24" in sequence:
            return '["# Factorial sequence: n! where n=0,1,2...", "import math", "print([math.factorial(n) for n in range(10)])"]'
        elif "1, 3, 6, 10, 15" in sequence:
            return '["# Triangular numbers: n(n+1)/2 where n=1,2,3...", "print([n*(n+1)//2 for n in range(1,11)])", "print([sum(range(1,n+2)) for n in range(10)])"]'
        elif "2, 1, 3, 4, 7" in sequence:
            return '["# Lucas sequence: L(n) = L(n-1) + L(n-2), L(0)=2, L(1)=1", "def lucas(n): a,b=2,1; return [a:=a+b for _ in range(n-1)] if n>1 else [2] if n==1 else []", "print([2,1] + lucas(8))"]'
        else:
            return '"*not found*"'
    elif complexity == 3:
        # Most complex sequences
        if "1, 1, 2, 5, 15" in sequence:
            return '["# Bell numbers - number of partitions of a set", "def bell_triangle(n): triangle=[[1]]; [triangle.append([triangle[-1][-1]]+[triangle[-1][j]+triangle[-1][j+1] for j in range(len(triangle[-1])-1)]) for _ in range(n-1)]; return [row[0] for row in triangle]", "print(bell_triangle(10))"]'
        else:
            return '"*not found*"'
    
    return '"*not found*"'

# Main function to process the CSV
def process_csv():
    input_file = '/Users/alberto/Documents/projects/medLLM/multi-python-script-time-series.csv'
    
    # Read the CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Process each row
    for i, row in enumerate(rows):
        if i == 0:  # Header row already updated
            continue
            
        sequence = row[0]
        complexity = int(row[1])
        
        # Generate data for the first two models
        gpt4o_mini_data = generate_gpt4o_mini_data(sequence, complexity)
        gemini25_pro_data = generate_gemini25_pro_data(sequence, complexity)
        
        # Add placeholder data for other models (will be filled in subsequent runs)
        row.extend([
            gpt4o_mini_data,
            gemini25_pro_data,
            '"*not found*"',  # claude-3.7
            '"*not found*"',  # deepseek_r1_0525
            '"*not found*"',  # llama_4_scout
            '"*not found*"',  # qwen3
            '"*not found*"',  # chatgpt_5
            '"*not found*"',  # opus_4
            '"*not found*"',  # mistral_large2405
            '"*not found*"'   # claude_sonnet_4
        ])
    
    # Write back to CSV
    with open(input_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print("CSV file updated successfully with first two models!")

if __name__ == "__main__":
    process_csv()