import time
from matplotlib import pyplot as plt
import numpy as np

def findNextCellToFill(grid, i, j):
    for x in range(i, 9):
        for y in range(j, 9):
            if grid[x][y] == 0:
                return x, y
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1


def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 * (i // 3), 3 * (j // 3)  # floored quotient should be used here.
            for x in range(secTopX, secTopX + 3):
                for y in range(secTopY, secTopY + 3):
                    if grid[x][y] == e:
                        return False
            return True
    return False


def solveSudoku(grid, i=0, j=0):
    i, j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1, 10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False



input1 = [[5,1,7,6,0,0,0,3,4],
         [2,8,9,0,0,4,0,0,0],
         [3,4,6,2,0,5,0,9,0],
         [6,0,2,0,0,0,0,1,0],
         [0,3,8,0,0,6,0,4,7],
         [0,0,0,0,0,0,0,0,0],
         [0,9,0,0,0,0,0,7,8],
         [7,0,3,4,0,0,5,6,0],
         [0,0,0,0,0,0,0,0,0]]

input2 = [
      [4, 5, 0, 8, 0, 0, 9, 0, 0],
      [0, 9, 0, 0, 5, 6, 0, 0, 4],
      [1, 0, 0, 0, 0, 0, 0, 0, 7],
      [2, 6, 0, 5, 4, 0, 0, 9, 0],
      [0, 0, 4, 1, 0, 2, 3, 0, 0],
      [0, 7, 0, 0, 6, 9, 0, 4, 8],
      [7, 0, 0, 0, 0, 0, 0, 0, 9],
      [8, 0, 0, 4, 9, 0, 0, 7, 0],
      [0, 0, 9, 0, 0, 3, 0, 2, 5]
    ]

input3 = [
      [3, 6, 0, 2, 0, 5, 0, 0, 0],
      [0, 1, 5, 4, 0, 3, 0, 8, 0],
      [0, 0, 4, 9, 1, 0, 0, 0, 0],
      [4, 5, 7, 0, 0, 0, 0, 9, 1],
      [0, 0, 2, 0, 0, 0, 3, 0, 0],
      [8, 3, 0, 0, 0, 0, 7, 6, 4],
      [0, 0, 0, 0, 9, 4, 8, 0, 0],
      [0, 2, 0, 3, 0, 6, 1, 4, 0],
      [0, 0, 0, 8, 0, 2, 0, 7, 9]
    ]

counts = [0,0,0]
for r in range(9):
    for c in range(9):
        if input1[r][c] == 0:
            counts[0] += 1
        if input2[r][c] == 0:
            counts[1] += 1
        if input3[r][c] == 0:
            counts[2] += 1

time_start = time.time()
solveSudoku(input1)
time_end = time.time()
times = [time_end-time_start]
time_start = time.time()
solveSudoku(input2)
time_end = time.time()
times += [time_end-time_start]
time_start = time.time()
solveSudoku(input3)
time_end = time.time()
times += [time_end-time_start]

for count in range(80):
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(count):
        row = np.random.randint(0,9)
        col = np.random.randint(0, 9)
        if input1[row][col] != 0:
            count1 += 1
        if input2[row][col] != 0:
            count2 += 1
        if input3[row][col] != 0:
            count3 += 1
        input1[row][col] = 0
        input2[row][col] = 0
        input3[row][col] = 0

    time_start = time.time()
    solveSudoku(input1)
    time_end = time.time()
    counts += [count1, count2, count3]
    times += [time_end - time_start]
    time_start = time.time()
    solveSudoku(input2)
    time_end = time.time()
    times += [time_end - time_start]
    time_start = time.time()
    solveSudoku(input3)
    time_end = time.time()
    times += [time_end - time_start]

for loop in range(5):
    count1 = 0
    count2 = 0
    count3 = 0
    for count in range(60,80):
        for i in range(count):
            row = np.random.randint(0,9)
            col = np.random.randint(0,9)
            if input1[row][col] != 0:
                count1 += 1
            if input2[row][col] != 0:
                count2 += 1
            if input3[row][col] != 0:
                count3 += 1
            input1[row][col] = 0
            input2[row][col] = 0
            input3[row][col] = 0
        time_start = time.time()
        solveSudoku(input1)
        time_end = time.time()
        counts += [count, count, count]
        times += [time_end - time_start]
        time_start = time.time()
        solveSudoku(input2)
        time_end = time.time()
        times += [time_end - time_start]
        time_start = time.time()
        solveSudoku(input3)
        time_end = time.time()
        times += [time_end - time_start]

plt.figure()
plt.plot(counts, times, 'r.')
plt.ylabel('Time to Solve')
plt.xlabel('Number of Unknowns')
plt.show()