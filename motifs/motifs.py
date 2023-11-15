from itertools import product

def GenerateMotifsByPattern(pattern, t):
  # Returns motifs according to given pattern
  L = len(pattern)
  idx = []
  idx.append(0)
  for i in range(L):
    idx.append(idx[len(idx) - 1] + pattern[i])
  motifs = []
  while idx[len(idx) - 1] != t + 1:
    motifs.append(idx.copy())
    for i in range(len(idx)):
      idx[i] += 1
  return motifs

def IteratePatterns(patterns, pattern, i, L, sum, Kmax):
  # Generate all patterns with sum <= Kmax and length = L - 1
  if i == L - 1:
    patterns.append(pattern.copy())
  else:
    for j in range(1, Kmax - sum + 1):
      pattern[i] = j
      IteratePatterns(patterns, pattern, i + 1, L, sum + j, Kmax)

# def GenPatterns(patterns, pattern, i, L, Kmax):
#   if i == L - 1:
#     patterns.append(pattern.copy())
#   else:
#     for j in range(1, Kmax + 1):
#       pattern[i] = j
#       GenPatterns(patterns, pattern, i + 1, L, Kmax)

def GenPatterns(L, x):
    elements = range(1, x+1)  # Создаем последовательность элементов от 1 до x
    sequences = product(elements, repeat=L)  # Генерируем все возможные комбинации длиной L
    return sequences


def GenerateAllMotifs(Kmax, L, t):
  print(1020030)
  # Returns map [pattern, [motifs...]]
  patterns = []
  pattern = []
  for i in range(L - 1):
    pattern.append(0)
  # IteratePatterns(patterns, pattern, 0, L, 0, Kmax)
  patterns = GenPatterns(L - 1, Kmax)
  # print(len(patterns))
  motifsByPatterns = []
  for p in patterns:
    motifs = GenerateMotifsByPattern(p, t)
    motifsByPatterns.append([p, motifs])
  return motifsByPatterns


# print(*GenPatterns(4, 10))

print(GenerateAllMotifs(10, 3, 40))
