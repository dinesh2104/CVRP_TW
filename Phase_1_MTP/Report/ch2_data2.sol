=== CVRP Instance ===
File: ch2_data2.txt
Depot: 0 at (0, 0)
Customers:
  1: (8, 0), demand = 2
  2: (2, 6), demand = 1
  3: (12, 0), demand = 1
  4: (10, 4), demand = 3
  5: (2, 10), demand = 1
  6: (12, 8), demand = 2
Vehicle capacity: 5

Solving CVRP using exponential algorithm...
Customers: [1, 2, 3, 4, 5, 6]
Capacity: 5
Demands: [2, 1, 1, 3, 1, 2]

New best solution found! Distance: 77.51
  Route 1: 0 -> 2 -> 3 -> 4 -> 0
    Demand: 5/5, Distance: 33.23
  Route 2: 0 -> 1 -> 5 -> 6 -> 0
    Demand: 5/5, Distance: 44.28

New best solution found! Distance: 70.57
  Route 1: 0 -> 2 -> 3 -> 4 -> 0
    Demand: 5/5, Distance: 33.23
  Route 2: 0 -> 1 -> 6 -> 5 -> 0
    Demand: 5/5, Distance: 37.34

New best solution found! Distance: 68.38
  Route 1: 0 -> 2 -> 4 -> 3 -> 0
    Demand: 5/5, Distance: 31.04
  Route 2: 0 -> 1 -> 6 -> 5 -> 0
    Demand: 5/5, Distance: 37.34

New best solution found! Distance: 63.77
  Route 1: 0 -> 1 -> 4 -> 0
    Demand: 5/5, Distance: 23.24
  Route 2: 0 -> 2 -> 5 -> 6 -> 3 -> 0
    Demand: 5/5, Distance: 40.52

Total solutions checked: 1129
=== OPTIMAL SOLUTION ===
Minimum total distance: 63.77
Number of vehicles used: 2

Route details:
Vehicle 1: 0 -> 1 -> 4 -> 0
  Demand: 5/5
  Distance: 23.24
Vehicle 2: 0 -> 2 -> 5 -> 6 -> 3 -> 0
  Demand: 5/5
  Distance: 40.52

Total demand served: 10
Total distance: 63.77
