# n_body_simulation

This is an optimized version of this code： https://github.com/pmocz/nbody-python

I have mainly addressed the following issues:

(1) Change the initial position and velocity distribution of stars to Gaussian distribution instead of random distribution.

(2) The dispersion of velocity distribution is determined by Virial law, and the dispersion of radius distribution is 1pc.

(3) The leap frog method is used to integrate the speed and position of each step, ensuring integration accuracy and total energy conservation of the system.

(4) Due to the non infinite proximity of two stars, a concept of "force softening" needs to be introduced.

(5) Explored whether any stars were kicked out of the system if the evolution time was long enough.

The evolution result is made into an animation and the total energy, total kinetic energy, and total potential energy of the entire system are calculated over time.
