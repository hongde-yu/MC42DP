import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import random, time, sys 

# Set up parameters
# T = 1500      # Temperature in K
kB = 0.00008617 # Boltzmann constant in eV/K
J = 0.0592    # J in meV
#J = 50e-3    # J in meV


# Set up lattice
L = 40 # Linear size of lattice
N = L*L # Total number of lattice sites

# Define function to calculate energy of a given spin configuration
def energy(lattice):
    E = 0
    for i in range(L):
        for j in range(L):
            s = lattice[i,j]
            if (i%2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                neighbors = lattice[(i-1)%L,j] + lattice[(i+1)%L,j] + lattice[i,(j-1)%L] # Calculate sum of neighbors' spins
            else:
                neighbors = lattice[(i-1)%L,j] + lattice[(i+1)%L,j] + lattice[i,(j+1)%L] # Calculate sum of neighbors' spins
            E += -J*s*neighbors
    return E/2

# Define function to calculate magnetization of a given spin configuration
#def magnetization(lattice):
#    return np.sum(lattice)

# Define function to perform a Monte Carlo step
def mc_step(lattice, beta):
    np.random.seed()
    i, j = np.random.randint(L, size=2) # Choose random lattice site
    s = lattice[i,j] # Get current spin at that site
    if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
        neighbors = lattice[(i-1)%L,j] + lattice[(i+1)%L,j] + lattice[i,(j-1)%L] # Calculate sum of neighbors' spins
    else:
        neighbors = lattice[(i-1)%L,j] + lattice[(i+1)%L,j] + lattice[i,(j+1)%L] # Calculate sum of neighbors' spins
    deltaE = 2*J*s*neighbors # Calculate change in energy if spin is flipped
    if deltaE <= 0 : # Decide whether to flip spin
        lattice[i,j] = -s # Flip spin
        return [deltaE, -2*s]
    elif np.exp(-beta*deltaE) > np.random.rand():
        lattice[i,j] = -s # Flip spin
        return [deltaE, -2*s]
    else:
        return [0.0 ,0.0]

# Define function to perform a Monte Carlo simulation
def mc_simulation(beta, num_steps, traj):
    np.random.seed()
    lattice = np.random.choice([-1/2, 1/2], size=(L,L)) # Initialize random spin configuration
    E_vals = np.zeros(num_steps) # Initialize array to store energy values
    M_vals = np.zeros(num_steps) # Initialize array to store magnetization values
#    E2_vals = np.zeros(num_steps) # Initialize array to store magnetization values
    E_now = energy(lattice)
    M_now = np.sum(lattice)
#    time0 = time.time()
    for i in range(num_steps):
#        if i%1000000 == 0:
#            time1 = time.time()
#            with open(str(traj)+".log", "a+") as f:
#                f.write("T{:.3f} t{} i{} {:.2f}\n".format(1/(beta*kB), traj, i , time1-time0))  
#            time0 = time1
        dE, dM = mc_step(lattice, beta) # Perform a Monte Carlo step
#       E_now = energy(lattice)
        E_now = E_now + dE 
        M_now = M_now + dM 
        E_vals[i] = E_now # Record energy after step
#        E2_vals[i] = E_now**2 # Record energy^2 after step
        M_vals[i] = M_now # Record magnetization after step
#    if (E_now - energy(lattice)) / E_now >0.0000001:
#        with open("error.txt", "a+") as f:
#            f.write("{} {} {} {}\n".format(traj, 1/(beta*kB), E_now, energy(lattice)) )
#    print ("M compare", M_now,np.sum(lattice)) 
#    with open("done.txt", "a+") as f:
#        f.write("{} {} \n".format(1/(beta*kB), traj ))  
    return E_vals, M_vals

def cal(T, traj):
    beta = 1/(kB*T)
    # Run Monte Carlo simulation
    num_steps = 50000000
    E_vals, M_vals = mc_simulation(beta, num_steps, traj)
    E2_vals = E_vals **2

    # Calculate average magnetization
    n_sample = -5000000
    M_avg = abs(np.mean(M_vals[n_sample:]))/N
    E_avg = np.mean(E_vals[n_sample:])
    E2_avg = np.mean(E2_vals[n_sample:])
    Cv = (E2_avg - E_avg**2) / (kB*T**2)
    print (M_avg, Cv)

    filename = "fig/%.4f_%d_%d_%.3f" %(T, L, traj, M_avg)
    # Plot energy and magnetization vs. time
#    plt.plot(range(num_steps), E_vals)
#    plt.xlabel('Time')
#    plt.ylabel('Energy')
#    plt.savefig(filename+"_energy.png")
#    plt.close()
    # plt.show()

#    plt.plot(range(num_steps), M_vals)
#    plt.xlabel('Time')
#    plt.ylabel('Magnetization')
#    plt.savefig(filename+"_mag.png")
#    plt.close()
    # plt.show()

    # Plot final state of lattice
#    plt.imshow(lattice, cmap='gray', vmin=-1, vmax=1)
#    plt.title(f'Magnetization = {M_avg:.2f}')
#    plt.axis('off')
#    plt.savefig(filename+"_status.png")
#    plt.close()
    # plt.show()
   
    del(M_vals, E_vals, E2_vals )
    return M_avg, Cv

##### Test  ##########
#time0 = time.time()
#cal(1.7, 1)
#print("time", time.time()-time0)
#sys.exit()

if __name__ == "__main__":
    Tem = float(sys.argv[1])/1.0
    n_traj = 50

    print ("------------------")   
    results = []
    pool = Pool(processes=50)
    for i in range(n_traj):
        results.append(pool.apply_async(cal, args = (Tem,i,)))
        # Ms[i] = cal(1500)
        # print(i, Ms[i])
    # print (np.mean(Ms))
    pool.close()
    pool.join()
    print ("Sub-process(es) done.")

    Ms = np.zeros(n_traj)
    Cvs = np.zeros(n_traj)
    i = 0
    for res in results:
        Ms[i], Cvs[i] = res.get()
        i += 1
    print ("TEMP: ", Tem)   
    print ("Mean: ", np.mean(Ms))   
    print ("------------------")  
    with open("results.txt", "a+") as f:
        f.write("{} {} {}\n".format(Tem, np.mean(Ms), np.mean(Cvs)) ) 


