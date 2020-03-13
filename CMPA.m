% Liam Anderson
% 100941879
% PA CMPA

clear all
clc

% 1. Generate Data
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

V = linspace(-1.95,0.7,200);

% Current
current = @(V) Is.*(exp(1.2/(25e-3).*V)-1)+ Gp.*V - Ib.*(exp(1.2/0.025.*(-(V+Vb))-1));
I1 = current(V);

% Current with experimental noise
I_N = I1.*(0.9+0.2*rand(1,length(I1)));


% 2. Polynomial Fitting
fit1 = polyfit(V,I1,4);
fit2 = polyfit(V,I1,8);
fit3 = polyfit(V,I_N,4);
fit4 = polyfit(V,I_N,8);

fitted_no_noise_4 = polyval(fit1,V);
fitted_noise_4 = polyval(fit2,V);
fitted_no_noise_8 = polyval(fit3,V);
fitted_noise_8 = polyval(fit4,V);


% 3. Nonlinear Curve Fitting

fo = fittype(@(A,B,C,D,x) (A.*exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1));
ff3 = fit(V', I_N', fo);
If3 = ff3(V);


% 4. Neural Net Model
inputs = V;
targets = I_N;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;


% PLOTS

% Polyfit
figure(1)
plot(V,I_N);
hold on;
plot(V,fitted_noise_4);
plot(V,fitted_noise_8);
hold off;
title('polyfit')

% Polyfit semilog
figure(2)
semilogy(V,abs(fitted_noise_4));
hold on;
semilogy(V,abs(fitted_noise_8));
hold off;
title('polyfit semilog')

% Nonlinear
figure(3)
plot(V,I_N);
hold on;
%hold on;
%plot(V,If1);
%plot(V,If2);
plot(V,If3);
title('nonlinear');

% Nonlinear semilog
figure(4)
% semilogy(V,abs(If1));
% semilogy(V,abs(If2));
%semilogy(V,abs(I_N));
%hold on;
semilogy(V,abs(If3));
%hold off;
title('nonlinear semilog');

% Neural Net
figure(5)
title('neural net');
plot(V,Inn);

% Neural Net semilog
figure(6)
semilogy(V,abs(Inn));
hold on;
semilogy(V,abs(I_N));
hold off;
title('neural net semilog');



