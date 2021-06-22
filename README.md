# Zynq706-CNN-Demod-Prediction
This MATLAB script(s) implement a CNN for demodulating different signals on the Zynq706 FPGA.
Currently the folder is a bit of a mess. The "main" file is labeled radarDemodCNN.m

Here you can select paremeters and run in a section-by-section mode based on needs.
Essentially, this program takes the example for CNN demodulation found here:
  https://www.mathworks.com/help/phased/ug/modulation-classification-of-radar-and-communication-waveforms-using-deep-learning.html
  
And adjusts it to be implented on the Zynq706 board. This will have the board run the CNN (which is a squeezenet) and return the results in addition to it's timings. A new repository will hopefully be created where the Zynq706 can sample an incoming signal, predict the demodulation type, and begin demodulating it all without the need for computer interaction (using the ARM processor).
