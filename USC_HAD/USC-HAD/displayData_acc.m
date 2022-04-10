%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Mi Zhang
%Date: July, 2010
%File Name: displayData_acc.m
%Description: Visualize the 3-axis accelerometer data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Fs = 100;               % Sampling frequency = 100Hz

% Read the data
data = sensor_readings;
acc_x = data(: ,1);
acc_y = data(: ,2);
acc_z = data(: ,3);

% Parameter declaration
BIN_COUNT = 10;         % 
T = 1/Fs;               % 
Interval = 1000 / Fs;   %
L = size(acc_x, 1);     % 
t = (0:L-1)*T;          %

% Visualize the time series, histogram, and FFT
figure;
% Time series
subplot(3, 3, 1);
plot(t, acc_x);
grid on;
xlabel('Time (s)');
ylabel('Acceleration (g)');
title('X-Axis Data');

% Histogram
subplot(3, 3, 2);
hist(acc_x, BIN_COUNT);
grid on;
xlabel('Acceleration (g)');
ylabel('Count');
title('X-Axis Distribution');

% Spectral analysis
subplot(3, 3, 3);
NFFT = 2^nextpow2(L);               
Y = fft(acc_x, NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
plot(f,2*abs(Y(1:NFFT/2+1)));
title('X-Axis Spectrum');
xlabel('Frequency (Hz)');
ylabel('|X(f)|');

% Time series
subplot(3, 3, 4);
plot(t, acc_y);
grid on;
xlabel('Time (s)');
ylabel('Acceleration (g)');
title('Y-Axis Data');

% Histogram
subplot(3, 3, 5);
hist(acc_y, BIN_COUNT);
grid on;
xlabel('Acceleration (g)');
ylabel('Count');
title('Y-Axis Distribution');

% Spectral analysis
subplot(3, 3, 6);
NFFT = 2^nextpow2(L);              
Y = fft(acc_y, NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
plot(f,2*abs(Y(1:NFFT/2+1)));
title('Y-Axis Spectrum');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

% Time series
subplot(3, 3, 7);
plot(t, acc_z);
grid on;
xlabel('Time (s)');
ylabel('Acceleration (g)');
title('Z-Axis Data');

% Histogram
subplot(3, 3, 8);
hist(acc_z, BIN_COUNT);
grid on;
xlabel('Acceleration (g)');
ylabel('Count');
title('Z-Axis Distribution');

% Spectral analysis
subplot(3, 3, 9);
NFFT = 2^nextpow2(L);               
Y = fft(acc_z, NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
plot(f,2*abs(Y(1:NFFT/2+1)));
title('Z-Axis Spectrum');
xlabel('Frequency (Hz)');
ylabel('|Z(f)|');


