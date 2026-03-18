#!/usr/bin/env python3
import random
import time

# ==========================================
# 1. INTERACTIVE SETUP 
# ==========================================
print("--- SENSORY PANEL BLOCK DESIGN GENERATOR ---")

try:
    NUM_BRANDS = int(input("Enter the TOTAL number of brands/samples being tested: "))
    NUM_TASTERS = int(input("Enter the TOTAL number of tasters: "))
    CUPS_PER_TASTER = int(input("Enter the number of samples EACH taster will evaluate: "))
except ValueError:
    print("\nERROR: Please enter whole numbers only (e.g., 10, not 'ten'). Run the script again.")
    exit()

# Safety check
if CUPS_PER_TASTER > NUM_BRANDS:
    print("\nERROR: A taster cannot evaluate more samples than the total number of brands!")
    exit()

# Target perfect numbers
expected_count = (NUM_TASTERS * CUPS_PER_TASTER) / NUM_BRANDS
expected_pairs = (expected_count * (CUPS_PER_TASTER - 1)) / (NUM_BRANDS - 1) if NUM_BRANDS > 1 else 0

print(f"\nRunning High-Speed Optimizer for {NUM_TASTERS} tasters...")
start_time = time.time()

# 1. Initialize random design
design = [random.sample(range(NUM_BRANDS), CUPS_PER_TASTER) for _ in range(NUM_TASTERS)]

# 2. Pre-calculate current counts and pairs
counts = [0] * NUM_BRANDS
pairs = [[0] * NUM_BRANDS for _ in range(NUM_BRANDS)]

for block in design:
    for i in range(len(block)):
        counts[block[i]] += 1
        for j in range(i + 1, len(block)):
            pairs[block[i]][block[j]] += 1
            pairs[block[j]][block[i]] += 1

# 3. Calculate initial Sum of Squared Errors (SSE)
def calculate_sse(current_counts, current_pairs):
    score = 0
    for c in current_counts:
        score += (c - expected_count) ** 2 * 10 # Penalty for uneven serving counts
    for i in range(NUM_BRANDS):
        for j in range(i + 1, NUM_BRANDS):
            score += (current_pairs[i][j] - expected_pairs) ** 2
    return score

best_score = calculate_sse(counts, pairs)
best_design = [list(b) for b in design]

# 4. Rapid Swapping Engine (Delta Evaluation)
for _ in range(150000):
    if best_score < 5:  # Stop early if we hit near-perfect mathematical balance
        break
        
    block_idx = random.randint(0, NUM_TASTERS - 1)
    item_in = random.randint(0, NUM_BRANDS - 1)
    
    if item_in not in design[block_idx]:
        item_out_idx = random.randint(0, CUPS_PER_TASTER - 1)
        item_out = design[block_idx][item_out_idx]
        
        # --- DELTA UPDATE ---
        counts[item_out] -= 1
        for other_item in design[block_idx]:
            if other_item != item_out:
                pairs[item_out][other_item] -= 1
                pairs[other_item][item_out] -= 1
                
        counts[item_in] += 1
        for other_item in design[block_idx]:
            if other_item != item_out:
                pairs[item_in][other_item] += 1
                pairs[other_item][item_in] += 1
        
        new_score = calculate_sse(counts, pairs)
        
        if new_score <= best_score or random.random() < 0.001:
            best_score = new_score
            design[block_idx][item_out_idx] = item_in
            best_design = [list(b) for b in design]
        else:
            # REVERT THE MATH 
            counts[item_in] -= 1
            for other_item in design[block_idx]:
                if other_item != item_out:
                    pairs[item_in][other_item] -= 1
                    pairs[other_item][item_in] -= 1
                    
            counts[item_out] += 1
            for other_item in design[block_idx]:
                if other_item != item_out:
                    pairs[item_out][other_item] += 1
                    pairs[other_item][item_out] += 1

print(f"Finished in {time.time() - start_time:.2f} seconds!")
print(f"Final Balance Score: {best_score:.2f} (Lower is better)")

# ==========================================
# 4. SPREADSHEET-READY OUTPUT (TSV)
# ==========================================
print("\n--- COPY BELOW THIS LINE ---")

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
header = "Taster\t" + "\t".join([f"Sample {i+1}" for i in range(CUPS_PER_TASTER)])
print(header)

for i, block in enumerate(best_design):
    row_data = f"{i+1}\t" + "\t".join([letters[x] for x in block])
    print(row_data)

print("--- COPY ABOVE THIS LINE ---")
