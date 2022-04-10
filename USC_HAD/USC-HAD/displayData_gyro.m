%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Author: Mi Zhang
%Date: July, 2010
%File Name: displayData_gyro.m
%Description: Visualize the 3-axis gyroscope data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Fs = 100;               % Sampling frequency = 100Hz

% Read the data
data = sensor_readings;
gyro_x = data(: ,4);
gyro_y = data(: ,5);
gyro_z = data(: ,6);

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
plot(t, gyro_x);
grid on;
xlabel('Time (s)');
ylabel('Gyro (dps)');
title('X-Axis Gyro Data');

% Histogram
subplot(3, 3, 2);
hist(gyro_x, BIN_COUNT);
grid on;
xlabel('Gyro (dps)');
ylabel('Count');
title('X-Axis Gyro Distribution');

% Spectral analysis
subplot(3, 3, 3);
NFFT = 2^nextpow2(L);               
Y = fft(gyro_x, NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
plot(f,2*abs(Y(1:NFFT/2+1)));
title('X-Axis Gyro Spectrum');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

% Time series
subplot(3, 3, 4);
plot(t, gyro_y);
grid on;
xlabel('Time (s)');
ylabel('Gyro (dps)');
title('Y-Axis Gyro Data');

% Histogram
subplot(3, 3, 5);
hist(gyro_y, BIN_COUNT);
grid on;
xlabel('Gyro (dps)');
ylabel('Count');
title('Y-Axis Gyro Distribution');

% Spectral analysis
subplot(3, 3, 6);
NFFT = 2^nextpow2(L);              
Y = fft(gyro_y, NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
plot(f,2*abs(Y(1:NFFT/2+1)));
title('Y-Axis Gyro Spectrum');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');

% Time series
subplot(3, 3, 7);
plot(t, gyro_z);
grid on;
xlabel('Time (s)');
ylabel('Gyro (dps)');
title('Z-Axis Gyro Data');

% Histogram
subplot(3, 3, 8);
hist(gyro_z, BIN_COUNT);
grid on;
xlabel('Gyro (dps)');
ylabel('Count');
title('Z-Axis Gyro Distribution');

% Spectral analysis
subplot(3, 3, 9);
NFFT = 2^nextpow2(L);               
Y = fft(gyro_z, NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2+1);
plot(f,2*abs(Y(1:NFFT/2+1)));
title('Z-Axis Gyro Spectrum');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');


