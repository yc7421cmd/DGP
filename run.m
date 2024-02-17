% 读取二进制音频文件
pos = 400;
los = cell(10);
k = 1;
while(pos <= 4000)
    filename = sprintf('data/%draw.mat', pos);
    binaryData = load(filename);  % 这里假设你的二进制数据保存在 .mat 文件中
    
    % 将二进制数据转换为音频信号
    audioSignal = double(binaryData.data);
    % 设置阈值，用于确定峰值
    threshold = 0.4*max(audioSignal);
    
    % 设置固定的时长
    fixedDuration = 1;  % 以秒为单位
    locations = [];
    % 计算每段信号应有的样本数
    samplesPerSegment = round(fixedDuration * 512);  
    i = 1;
    while i < length(audioSignal)
        if(abs(audioSignal(i,1)) >= threshold)
            locations = [locations;i];
            i = i + 200;
        end
        i = i + 1;
    end
    % 分离出的每段音频时长相等
    separatedSignals = cell(length(locations) - 1, 1);
    
    for i = 1:length(locations) - 1
        startSample = ceil(locations(i) - samplesPerSegment/4);  %以peak为开始
        endSample = ceil(locations(i) + samplesPerSegment*3/4); 
        % 确保不超过信号的末尾
        if endSample > length(audioSignal)
            break;
        end
        
        separatedSignals{i} = audioSignal(startSample:endSample);
    end
    los{k} = separatedSignals{1};
    %绘制频谱
%     subplot(5, 2, k);
%     for i = 1:length(separatedSignals)
%         segment = separatedSignals{i};
%         fftResult = fft(segment);
%         
%         % 计算频谱的振幅
%         amplitude = abs(fftResult);
%         
%         % 计算对应的频率
%         fs = 51200;  % 采样率，根据实际情况调整
%         f = (0:length(segment)-1) * fs / length(segment);
%         
%         % 绘制频谱图，使用 hold on 保持当前图形
%         hold on;
%         plot(f, amplitude, 'DisplayName', ['Signal ' num2str(i)]);
%     end
%     xlabel('Frequency (Hz)');
%     ylabel('P1(f)');
%     title(sprintf('Spectrum of Separated %d Signals', pos));

    %听一下每次的敲击信号
    % for i = 1:length(separatedSignals)
    %     sound(separatedSignals{i}, 4410);
    %     pause(4);  % 可以根据需要调整每段音频之间的暂停时间
    % end
    
    % 播放音频
    % sound(audioSignal, 51200);
    pos = pos + 400;
    k = k + 1;
end
% 绘制第一次敲击的时域图像

for i = 1:10
    subplot(5, 2, i);
    time = (0:length(los{i}) - 1) / 51.2;  % 假设采样率为 51200 Hz
    plot(time, los{i});
    xlabel('time（ms）');
    ylabel('amplitude');
    m = 400 * i;
    title(sprintf('松紧度为%d的第一次敲击时域图像', m));
end




