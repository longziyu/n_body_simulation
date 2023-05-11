import numpy as np
import matplotlib.pyplot as plt

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

def getAcc( pos, mass, G, softening ):
    """
    Calculate the acceleration on each particle due to Newton's Law
    pos is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """

    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
    inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax,ay,az))

    return a

def getEnergy( pos, vel, mass, G ):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos is N x 3 matrix of positions
    vel is N x 3 matrix of velocities
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    KE is the kinetic energy of the system
    PE is the potential energy of the system
    """

    # Kinetic Energy:
    KE = 0.5 * np.sum(np.sum( mass * vel**2 ))

    # positions r = [x,y,z] for all particles
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

    # sum over upper triangle, to count each interaction only once
    PE = G * np.sum(np.sum(np.triu(-(mass*mass.T)*inv_r,1)))

    return KE, PE;

def main():
    """ N-body simulation """

    # Simulation parameters
    N = 100 # Number of particles
    t = 0 # current time of the simulation
    tEnd = 10.0 # time at which simulation ends
    dt = 0.01 # timestep
    softening = 0.1 # softening length
    G = 1.0 # Newton's Gravitational Constant
    plotRealTime = True # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(17) # set the random number generator seed

    # Set up positions and velocities as Gaussian distributions
    pos_mean = np.zeros(3)
    pos_cov = np.identity(3) * 1.0 # radius distribution has a dispersion of 1pc
    pos = np.random.multivariate_normal(pos_mean, pos_cov, size=N)

    vel_mean = np.zeros(3)
    vel_cov = np.identity(3) * np.sqrt(G * 20.0 / 1.0) # use Virial theorem to determine velocity dispersion
    vel = np.random.multivariate_normal(vel_mean, vel_cov, size=N)
    mass = 20.0*np.ones((N,1))/N
    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel,0) / np.mean(mass)

    # Calculate initial gravitational accelerations
    acc = getAcc( pos, mass, G, softening )

    # Calculate initial energy of system
    KE, PE = getEnergy( pos, vel, mass, G )

    # Number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # Save energies, particle orbits for plotting trails
    pos_save = np.zeros((N,3,Nt+1))
    pos_save[:,:,0] = pos
    KE_save = np.zeros(Nt+1)
    KE_save[0] = KE
    PE_save = np.zeros(Nt+1)
    PE_save[0] = PE
    t_all = np.arange(Nt+1)*dt

    # Prepare figure
    fig = plt.figure(figsize=(8,8), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])

    # Initialize counter variable for kicked-out stars
    kicked_out = 0

    # Simulation Main Loop
    for i in range(Nt):
        # Leap-frog Method
        vel_half = vel + acc * dt/2.0
        pos += vel_half * dt
        acc = getAcc( pos, mass, G, softening )
        vel = vel_half + acc * dt/2.0

        # Check if any star has been kicked out
        distance_from_com = np.sqrt(np.sum(pos**2, axis=1))
        kicked_out += np.sum(distance_from_com > 10.0)

        # Update time
        t += dt

        # Get energy of system
        KE, PE  = getEnergy( pos, vel, mass, G )

        # Save energies, positions for plotting trail
        pos_save[:,:,i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE

        # Print to screen
        print("t = ", t, "KE = ", KE, "PE = ", PE, "E = ", KE+PE, "Kicked out: ", kicked_out)

        # Plot positions
        if (plotRealTime):
            ax1.clear()
            ax1.scatter(pos[:,0], pos[:,1], s=0.5)
            ax1.set_xlim(-5,5)
            ax1.set_ylim(-5,5)
            ax1.set_aspect('equal', 'box')
            ax1.set_title('t = {:.2f}'.format(t))
            ax2.clear()
            ax2.plot(t_all[:i+1], KE_save[:i+1], 'b', label='KE')
            ax2.plot(t_all[:i+1], PE_save[:i+1], 'r', label='PE')
            ax2.plot(t_all[:i+1], KE_save[:i+1]+PE_save[:i+1], 'g', label='E')
            ax2.legend(loc='upper right')
            plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    main()
