%% parameters
subject = 'A';
model_name = 'xDAWN+Riemann+LR';
time_long = 600;  %ms
Fs = 240;  %Hz
channel = 11;  % Cz
fontsize = 15;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% prepare
model_dir = ['../../runs/' model_name '/' subject];
epoch_file = [model_dir '/' 'avg_epochs_real.mat'];

% figure settings.
if strcmp(subject, 'A')
    y_limit = [-0.15, 0.25];
    tf_c_limit = [0, 2.7e-3];
    topo_c_limit = [0 0.08];
elseif strcmp(subject, 'B')
    y_limit = [-0.15, 0.32];
    tf_c_limit = [0, 3.6e-3];
    topo_c_limit = [0 0.12];
else
    error('The parameter \"subject\" should be either \"A\" or \"B\"!');
end

load(epoch_file);

%% averaging on all the trials
clean_target = squeeze(mean(clean_target, 1));
clean_nontarget = squeeze(mean(clean_nontarget, 1));
adv_target = squeeze(mean(adv_target, 1));
adv_nontarget = squeeze(mean(adv_nontarget, 1));

clean_target = permute(clean_target, [2, 1]);
clean_nontarget = permute(clean_nontarget, [2, 1]);
adv_target = permute(adv_target, [2, 1]);
adv_nontarget = permute(adv_nontarget, [2, 1]);

time_length = size(clean_target, 1);
time_axis = (0:1:time_length-1)/Fs;

%% plot averaged responses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
subplot(121);
plot(time_axis,clean_target(:,channel),'linewidth',2)
hold on
plot(time_axis,clean_nontarget(:,channel),'r','linewidth',2)
ylim(y_limit)
legend('Targets','Nontargets');
xlabel('Time (s)');
ylabel('Amplitude');
title('Clean');
set(gca, 'fontsize',fontsize);

subplot(122);
plot(time_axis,adv_target(:,channel),'linewidth',2);
hold on
plot(time_axis,adv_nontarget(:,channel),'r','linewidth',2)
ylim(y_limit)
legend('Targets','Nontargets');
xlabel('Time (s)');
ylabel('Amplitude');
title('Adversarial');
set(gcf,'name','Averaged P300 Responses over Cz');
set(gca, 'fontsize',fontsize);
set (gcf,'Position',[100,100,800,300], 'color','w')
saveas(gcf, [subject '_P300' '.eps'], 'psc');

%%  spectrom %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target/NonTarget voltage spectrom
clean_diff = clean_target(:,channel);% - clean_nontarget(:, channel);
adv_diff = adv_target(:,channel);% - adv_nontarget(:, channel);
figure
subplot(121);
[~, F, T, P] = spectrogram(clean_diff, 32, 30, 240, Fs);
imagesc(T, F(1:30), abs(P(1:30, :)));
axis xy
xlabel('Time (s)'); ylabel('Frequency (Hz)');
caxis(tf_c_limit);
title('Clean');
set(gca, 'fontsize', fontsize);

subplot(122);
[~, F, T, P] = spectrogram(adv_diff, 32, 30, 240, Fs);
imagesc(T, F(1:30), abs(P(1:30, :)));
axis xy
xlabel('Time (s)'); ylabel('Frequency (Hz)');
caxis(tf_c_limit);
title('Adversarial');
set(gca, 'fontsize', fontsize);

axes('position', [0.8, 0.2, 0.2, 0.7]);
axis off;
caxis(tf_c_limit);
colorbar();

set(gcf, 'name', 'Target/Nontarget Difference Specotrogram');
set (gcf,'Position',[100,100,800,300], 'color','w');
saveas(gcf, [subject '_spectrogram' '.eps'], 'psc');

%%  topography %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target/NonTarget voltage topography plot at 300ms (sample 72)
clean_vdiff=abs(clean_target(72, :)-clean_nontarget(72, :));
figure
subplot(121);
topoplotEEG(clean_vdiff,'eloc64.txt','gridscale',150);
caxis(topo_c_limit);
title('Clean');
set(gca, 'fontsize', fontsize);
set(gca, 'position', [0.13, 0.05, 0.3347, 0.785]);

subplot(122);
adv_vdiff=abs(adv_target(72, :)-adv_nontarget(72, :));
topoplotEEG(adv_vdiff,'eloc64.txt','gridscale',150);
caxis(topo_c_limit);
title('Adversarial');
set(gca, 'fontsize', fontsize);
set(gca, 'position', [0.47, 0.05, 0.3347, 0.785]);

axes('position', [0.7, 0.15, 0.2, 0.7]);
axis off;
caxis(topo_c_limit);
colorbar();

set(gcf, 'name', 'Target/Nontarget Difference Topoplot');
set (gcf,'Position',[100,100,720,300], 'color','w');
saveas(gcf, [subject '_topoplot' '.eps'], 'psc')

