# With the predicted demand, demand is predicted demand, act_demand is actual demandimport gurobipy as gp
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Define parameters and data
stations = [245,291,185,146,178,190,243,288,161,148,267,376,320,299,426,558,353,221,281,118,177,360,826,799,359,108,646,
            228,226,325,64,83,318,388,129,338,335,192,383,386,260,174,256,594,283,18,372,109,244,358,751,809,562,15] # List of bike station IDs
cluster1 = [245,291,185,146,178,190,243,288,161,148,267,376,320,299,426,558,353,221,281,118,177,360,826,799,359,108,646]
cluster2 = [228,226,325,64,83,318,388,129,338,335,192,383,386,260,174,256,594,283,18,372,109,244,358,751,809,562,15]

constraints = [29, 32, 14, 35, 19, 21, 25, 34, 16, 16, 22, 24, 26, 19, 17, 24, 17, 
               16, 18, 13, 25, 27, 28, 24, 18, 29, 16, 40, 24, 18, 26, 21, 18, 
               14, 18, 24, 33, 16, 18, 16, 18, 35, 15, 33, 16, 27, 24, 58, 18, 20, 
               24, 28, 21, 26]
capacity = {station: constraint for station, constraint in zip(stations, constraints)}
cost = pd.DataFrame(index=stations, columns=stations)
for i in stations:
    for j in stations:
        if i == j:
            cost.loc[i, j] = 0
        elif i in cluster1 and j in cluster1:
            cost.loc[i, j] = 1
        elif i in cluster1 and j in cluster2:
            cost.loc[i, j] = 2
        elif i in cluster2 and j in cluster1:
            cost.loc[i, j] = 2
        elif i in cluster2 and j in cluster2:
            cost.loc[i, j] = 1

cost = cost.astype(int)
print(cost)
 
def solve_opt( x, demand, supply, pattern, act_demand, rev = 5): # Note: easier to define as a function, because eventually you have to solve this for a lot of iterations, so all of the parameters that will change might as well be defined as a parameter to the function.
    # Creates a list of permissible decisions
    if (pattern): # pattern == 1 refers to across clusters
        paths = [ (i,j) for i in cluster1 for j in cluster2 ]
    else: # pattern == 0 refers to within clusters
        paths = [ (i,j) for i in cluster1 for j in cluster1 if i>j] # Note: i == j will end up repeating pairs
        paths += [ (i,j) for i in cluster2 for j in cluster2 if i > j]
 
    if isinstance(cost, pd.DataFrame):
        c = { (i,j) : cost.loc[i,j] for (i,j) in paths }
    else:
        c = { (i,j) : cost for (i,j) in paths } 
        # You could do the same thing for rev if you want to vary the revenue at different bike points

    m = gp.Model("BikeReloc")

    # Defines variables
    z = m.addVars( paths, lb = 0, ub = GRB.INFINITY, name = "z") # Once again, only adds variables that correspond to allowed paths
    y = m.addVars( stations, lb = 0, ub = GRB.INFINITY, name = "y") # These are the actual departures
    r = m.addVars( stations, lb = 0, ub = GRB.INFINITY, name = "w") # Revenue at each station
    # Sets objective
    m.setObjective( gp.quicksum( [r[i] for i in stations] )  - gp.quicksum( [ z[i, j]*c[i,j] for (i,j) in paths ] ), GRB.MAXIMIZE)
 
     # Constraint 1: End-state >= 0
    m.addConstrs((x[i] + gp.quicksum([z.get((j, i2), 0) for (i2, j) in paths if i2 == i and i2 != j]) - gp.quicksum([z.get((i2, j), 0) for (i2, j) in paths if i2 == i and i2 != j ]) + supply[i]  + y[i] >= 0 for i in stations), name="zero")

    # Constraint 2: End-state <= capacity
    m.addConstrs((x[i] 
                + gp.quicksum([z.get((j, i2), 0) for (i2, j) in paths if i2 == i and i2 != j])
                - gp.quicksum([z.get((i2, j), 0) for (i2, j) in paths if i2 == i and i2 != j])
                - y[i] 
                + supply[i]
                <= capacity[i] for i in stations), name="cap")

    # Constraints for y
    m.addConstrs( ( y[i] <= demand[i] for i in stations ), name = "y_demand")
 
    # Constraints for r
    m.addConstrs( ( r[i] <= act_demand[i]*rev for i in stations ), name = "r_act_demand")
    m.addConstrs( ( r[i] <= y[i]*rev for i in stations ), name = "r_demand")

    # Constraints for bike relocation
    m.addConstrs((gp.quicksum([z.get((j, i), 0) for (j, i2) in paths if i2 == i]) - gp.quicksum([z.get((i, j), 0) for (i2, j) in paths if i2 == i]) >= y[i] - supply[i] for i in stations), name="bike_relocation")

    # Optimize the model
    m.optimize()

    # Retrieve the optimal solution and objective value
    if m.status == GRB.OPTIMAL:
        solution = m.getAttr("x", z)
        objective_value = m.objVal
        
        # Loop over the decision variables
        for var in solution:
            # Print the pair of stations and the number of bikes
            if solution[var] > 0:
                print(f'Station {var[0]} transfers {solution[var]:.0f} bikes to station {var[1]}')

    else:
        solution = None
        objective_value = None
            
    return m.status, solution, objective_value

# With the predicted demand, demand is predicted demand, act_demand is actual demand
# Read the CSV file
df = pd.read_csv('TestingDataPrediction.csv')

# Create the demand dictionary
demand_dict = df.groupby('ID')['Demand'].apply(list).to_dict()
# Create the timepoints dictionary
timepoints_dict = df.groupby('ID')['Tf'].apply(list).to_dict()
# Create the initial bikes dictionary
bikes_dict = df.groupby('ID')['Bikes'].apply(list).to_dict()
# Create the capacity dictionary
capacity_dict = df.groupby('ID')['Supply'].apply(list).to_dict()
# Create the actual demand dictionary
act_demand_dict = df.groupby('ID')['Act_Demand'].apply(list).to_dict()
# Replace demand and timepoints with respective dictionaries
station_data = df.groupby('ID').first().reset_index()
station_data['Demand'] = station_data['ID'].map(demand_dict)
station_data['Tf'] = station_data['ID'].map(timepoints_dict)
station_data['Bikes'] = station_data['ID'].map(bikes_dict)
station_data['Supply'] = station_data['ID'].map(capacity_dict)
station_data['Act_Demand'] = station_data['ID'].map(act_demand_dict)

# Now we can add 'Station', 'Bikes', 'Demand', and 'T' to the dictionary 
my_dict = station_data.set_index('ID')[['Bikes', 'Demand', 'Tf', 'Supply','Act_Demand']].T.to_dict('list')
#my_dict = station_data.set_index('ID')[['Bikes', 'Demand', 'Tf', 'Supply']].T.to_dict('list')
# And modify the simulate function to extract single values instead of lists:
def simulate(my_dict, pattern):
    # Get unique time steps from the first station (assuming all have the same time steps)
    time_steps = range(len(list(my_dict.values())[0][2]))
    total_profit = 0
    total_cost_dict = {}
    for t in time_steps:
        print(f"\nTime {t}")

        # Initialize the state of each bike station
        state = {station: my_dict[station][0][t] for station in my_dict}
        #print(f"Starting state: {state}")

        # Get the demand for the current time step
        demand = {station: my_dict[station][1][t] for station in my_dict}
        #print(f"Demand: {demand}")
        supply = {station: my_dict[station][3][t] for station in my_dict}
        act_demand = {station: my_dict[station][4][t] for station in my_dict}
        #(f"Actual Demand: {act_demand}")
        # Optimize bike allocation
        status, solution, objective_value = solve_opt(x=state, demand=demand, supply=supply, pattern=pattern, act_demand=act_demand)
        
        # If the optimization was successful, update the state based on the solution
        if status == GRB.OPTIMAL:
            for (i, j) in solution:
                bikes_transferred = solution[(i, j)]
                # Remove transferred bikes from the origin station
                state[i] -= bikes_transferred
                # Add transferred bikes to the destination station
                state[j] += bikes_transferred
        
            # Update the initial number of bikes at each station based on the supply
            for station in state:
                state[station] += supply[station]
                
            # Subtract bikes used for trips from stations with demand
            for station in state:
                state[station] -= min(act_demand[station], state[station])
                #state[station] -= min(demand[station], state[station])
            total_cost = sum(solution[i, j] * cost[i][j] for (i, j) in solution)  # Calculate the total cost
            print(f"Total cost: {total_cost:.0f}")
            total_profit += objective_value
            print(f"Optimal solution: {objective_value}")
            #print(f"Ending state: {state}")
            total_cost_dict[my_dict[list(my_dict.keys())[0]][2][t]] = total_cost
        print(f"Total cost dictionary: {total_cost_dict}")    
        print(f"Total profit: {total_profit}")
    return total_cost_dict, total_profit

total_cost_dict, total_profit = simulate(my_dict=my_dict, pattern=0)

print(f"Total cost dictionary: {total_cost_dict}")