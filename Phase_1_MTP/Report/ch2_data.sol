=== CVRP Instance ===
File: ch2_data.txt
Depot: 0 at (0, 0)
Customers:
  1: (7, 1), demand = 2
  2: (3, 5), demand = 1
  3: (11, 10), demand = 1
  4: (9, 3), demand = 3
  5: (2, 9), demand = 1
  6: (11, 7), demand = 2
Vehicle capacity: 5

Solving CVRP using exponential algorithm...
Customers: [1, 2, 3, 4, 5, 6]
Capacity: 5
Demands: [2, 1, 1, 3, 1, 2]

New best solution found! Distance: 70.79
  Route 1: 0 -> 2 -> 3 -> 4 -> 0
    Demand: 5/5, Distance: 32.03
  Route 2: 0 -> 1 -> 5 -> 6 -> 0
    Demand: 5/5, Distance: 38.76

New best solution found! Distance: 64.75
  Route 1: 0 -> 2 -> 3 -> 4 -> 0
    Demand: 5/5, Distance: 32.03
  Route 2: 0 -> 1 -> 6 -> 5 -> 0
    Demand: 5/5, Distance: 32.72

New best solution found! Distance: 56.09
  Route 1: 0 -> 1 -> 4 -> 0
    Demand: 5/5, Distance: 19.39
  Route 2: 0 -> 2 -> 3 -> 6 -> 5 -> 0
    Demand: 5/5, Distance: 36.70

New best solution found! Distance: 54.43
  Route 1: 0 -> 1 -> 4 -> 0
    Demand: 5/5, Distance: 19.39
  Route 2: 0 -> 2 -> 5 -> 3 -> 6 -> 0
    Demand: 5/5, Distance: 35.05

Total solutions checked: 1129
=== OPTIMAL SOLUTION ===
Minimum total distance: 54.43
Number of vehicles used: 2

Route details:
Vehicle 1: 0 -> 1 -> 4 -> 0
  Demand: 5/5
  Distance: 19.39
Vehicle 2: 0 -> 2 -> 5 -> 3 -> 6 -> 0
  Demand: 5/5
  Distance: 35.05

Total demand served: 10
Total distance: 54.43
