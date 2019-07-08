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
epoch_file = [model_dir '/' 'avg_epochs.mat'];

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

subplot 231;
plot(time_axis,clean_target(:,channel),'linewidth',2)
hold on
plot(time_axis,clean_nontarget(:,channel),'r','linewidth',2)
ylim(y_limit)
legend('Targets','Nontargets');
xlabel('Time (s)');
ylabel({'{\bfClean}'; ''; 'Amplitude'});
title('Averaged Responses (Cz)');
set(gca, 'fontsize', fontsize);

subplot 234;
plot(time_axis,adv_target(:,channel),'linewidth',2);
hold on
plot(time_axis,adv_nontarget(:,channel),'r','linewidth',2)
ylim(y_limit);
legend('Targets','Nontargets');
xlabel('Time (s)');
ylabel({'{\bfAdversarial}'; ''; 'Amplitude'});
set(gca, 'fontsize',fontsize);

%%  spectrom %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target/NonTarget voltage spectrom
subplot 232;
clean_diff = clean_target(:,channel);% - clean_nontarget(:, channel);
adv_diff = adv_target(:,channel);% - adv_nontarget(:, channel);
[~, F, T, P] = spectrogram(clean_diff, 32, 30, 240, Fs);
imagesc(T, F(1:30), abs(P(1:30, :)));
axis xy;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
caxis(tf_c_limit);
colorbar;
title('Spectrogram (Cz)');
set(gca, 'fontsize', fontsize);

subplot 235;
[~, F, T, P] = spectrogram(adv_diff, 32, 30, 240, Fs);
imagesc(T, F(1:30), abs(P(1:30, :)));
axis xy;
xlabel('Time (s)'); ylabel('Frequency (Hz)');
caxis(tf_c_limit);
colorbar;
set(gca, 'fontsize', fontsize);

%%  topography %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target/NonTarget voltage topography plot at 300ms (sample 72)
clean_vdiff=abs(clean_target(72, :)-clean_nontarget(72, :));
subplot 233;
topoplotEEG(clean_vdiff,'eloc64.txt','gridscale',150);
caxis(topo_c_limit);
colorbar;
title('Topography (300ms)');
set(gca, 'fontsize', fontsize);

adv_vdiff=abs(adv_target(72, :)-adv_nontarget(72, :));
subplot 236;
topoplotEEG(adv_vdiff,'eloc64.txt','gridscale',150);
caxis(topo_c_limit);
colorbar;
set(gca, 'fontsize', fontsize);
set(gcf,'Position',get(0,'ScreenSize'));


