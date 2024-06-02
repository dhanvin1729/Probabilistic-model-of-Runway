# Probabilistic-model-of-Runway
This project was given to us by Group Captain Vinay Bhardwaj (Indian Air Force) and was completed under the guidance and supervision of Prof. Kuldeep Singh and Prof. Rajendra Mitharwal of MNIT Jaipur.

The problem statement of the project was to simulate missiles of certain specifications landing on a runway of given dimensions. This is a probabilistic model that uses the Gaussian probability density function to simulate the random nature of missiles and their predicted landing points on the runway.

The GUI, designed using Tkinter, allows the user to input desired values of runway dimensions, missile CEP(Circular Error Probable), missile damage radius etc. The simulation then runs on these dimensions.

The blue filled rectangle represents the parking or dispersal area alongside the runway, and the blue unfilled rectangle represents the runway. The user clicks on a certain point on the runway, marking the aimpoint of the first missile. The red circle around the aimpoint represents the CEP of the missile. The actual point of impact on the runway is determined by the Gaussian probability distribution function. The red circle represents the strikepoint of the missile. The green dots are the submunitions arising from the missile that land in an approximately uniform pattern over the submunition dispersal radius, which is shown by the green circle. Around 80-90% of the submunitions are successful, and the unsuccessful ones are randomly decided from the total no of submunitions.

The output in the GUI shows the no of successful submunitions, the lengths of vertical strips remaining after each missile, and the total number of missiles required in each runtime for the runway to be destroyed completely along its width.

