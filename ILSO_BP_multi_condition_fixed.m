%% ILSO-BP算法优化创新实现 - 多工况数据融合增强版 (修复版)
clear all; close all; clc;

% ================== 配置参数 ==================
numRuns =5;                % 运行次数
fixedSeed = 42;             % 固定随机种子（确保可复现性）
saveResults = true;         % 是否保存结果
showIndividualPlots = false;% 是否显示单次运行结果图
featureAnalysis = true;     % 是否进行特征分析可视化
window_size = 30;           % 滑动窗口大小 (新增)

% ================== 多工况数据加载 ==================
dataFiles = {
    'TX_6m.xlsx',   % 6m/min数据
    'TX_8m.xlsx',   % 8m/min数据
    'TX_10m.xlsx'   % 10m/min数据
};

allData = [];
allOutput = [];
speedLabels = []; % 存储速度标签 (新增)

% 加载并融合多工况数据
fprintf('加载多工况数据...\n');
for i = 1:length(dataFiles)
    excelFile = xlsread(dataFiles{i});
    
    % 提取温度测点数据作为输入
    rawData = excelFile(2:end, 3:8); % T2-T8列
    % 提取 X0 列作为输出
    outputData = excelFile(2:end, end);
    
    % 从文件名提取进给速度
    speed = str2double(regexp(dataFiles{i}, '\d+', 'match')); % 提取数字
    if isempty(speed)
        error('文件名中未检测到速度值: %s', dataFiles{i});
    end
    
    % 添加速度特征
    speedFeature = speed * ones(size(rawData, 1), 1);
    rawData = [rawData, speedFeature];
    
    % 累积数据
    allData = [allData; rawData];
    allOutput = [allOutput; outputData];
    speedLabels = [speedLabels; repmat(speed, size(rawData, 1), 1)]; % 新增
    fprintf('已加载: %s, 样本量: %d, 速度: %dm/min\n', dataFiles{i}, size(rawData, 1), speed);
end

% 计算总样本量
totalSamples = size(allData, 1);
fprintf('总样本量: %d (6m/min:%d, 8m/min:%d, 10m/min:%d)\n',...
    totalSamples,...
    sum(speedLabels == 6),...
    sum(speedLabels == 8),...
    sum(speedLabels == 10));

% ================== 特征工程增强 (修复数据泄露) ==================
fprintf('执行多工况特征工程...\n');

% 1. 基础特征：原始温度+速度
baseFeatures = allData;

% 2. 温度特征：温差和变化率
tempFeatures = allData(:, 1:end-1); % 去掉速度列
deltaT = [zeros(1, size(tempFeatures, 2)); diff(tempFeatures)];
tempRate = deltaT / 240; % 采样间隔240秒

% 3. 速度特征：多项式变换
speed = allData(:, end);
speedFeatures = [speed, speed.^2, sqrt(speed), log(speed+1)];

% 4. 交互特征：温度之间交互
interactionFeatures = [];
numTemp = size(tempFeatures, 2);
for i = 1:numTemp
    for j = i+1:numTemp
        interactionFeatures = [interactionFeatures, tempFeatures(:, i).*tempFeatures(:, j)];
    end
end

% 5. 温度-速度交互
tempSpeedInteraction = tempFeatures .* speed;

% 6. 时域特征 (滑动窗口计算)
fprintf('计算时域特征(滑动窗口:%d)...\n', window_size);
timeDomainFeatures = zeros(size(tempFeatures, 1), 4*numTemp);
for i = 1:numTemp
    col = tempFeatures(:, i);
    
    % 移动平均和标准差
    movAvg = movmean(col, window_size);
    movStd = movstd(col, window_size);
    
    % 滑动窗口计算偏度和峰度 (修复数据泄露)
    skewVals = zeros(size(col));
    kurtVals = zeros(size(col));
    for pos = 1:length(col)
        startIdx = max(1, pos - window_size + 1);
        window = col(startIdx:pos);
        if numel(window) >= 3 % 至少3个样本计算统计量
            skewVals(pos) = skewness(window);
            kurtVals(pos) = kurtosis(window);
        end
    end
    
    timeDomainFeatures(:, (i-1)*4+1) = movAvg;
    timeDomainFeatures(:, (i-1)*4+2) = movStd;
    timeDomainFeatures(:, (i-1)*4+3) = skewVals;
    timeDomainFeatures(:, (i-1)*4+4) = kurtVals;
end

% 7. 频域特征 (滑动窗口计算)
fprintf('计算频域特征(滑动窗口:%d)...\n', window_size);
freqDomainFeatures = zeros(size(tempFeatures, 1), 3*numTemp);
for i = 1:numTemp
    col = tempFeatures(:, i);
    for pos = window_size:length(col)
        window_data = col(pos-window_size+1:pos);
        L = length(window_data);
        F = fft(window_data);
        P2 = abs(F/L);
        P1 = P2(1:floor(L/2)+1);
        P1(2:end-1) = 2*P1(2:end-1);
        
        % 获取主要频率分量
        [~, idx] = sort(P1, 'descend');
        topFreqs = zeros(3, 1);
        for j = 1:min(3, length(P1))
            topFreqs(j) = P1(idx(j));
        end
        freqDomainFeatures(pos, (i-1)*3+1:(i-1)*3+3) = topFreqs';
    end
end

% 8. 特征命名 (新增)
featureNames = {};
for i = 1:6
    featureNames{end+1} = sprintf('T%d', i+1);
end
featureNames{end+1} = 'Speed';

% 添加温差和变化率
for i = 1:6
    featureNames{end+1} = sprintf('dT%d', i+1);
end
for i = 1:6
    featureNames{end+1} = sprintf('rateT%d', i+1);
end

% 添加交互特征
for i = 1:5
    for j = i+1:6
        featureNames{end+1} = sprintf('T%d*T%d', i+1, j+1);
    end
end

% 添加速度特征
speedNames = {'Speed', 'Speed^2', 'sqrt(Speed)', 'log(Speed+1)'};
featureNames(end+1:end+4) = speedNames;

% 添加温度-速度交互
for i = 1:6
    featureNames{end+1} = sprintf('T%d*Speed', i+1);
end

% 添加时域特征
for i = 1:6
    featureNames{end+1} = sprintf('T%d_MA', i+1);
    featureNames{end+1} = sprintf('T%d_Std', i+1);
    featureNames{end+1} = sprintf('T%d_Skew', i+1);
    featureNames{end+1} = sprintf('T%d_Kurt', i+1);
end

% 添加频域特征
for i = 1:6
    for j = 1:3
        featureNames{end+1} = sprintf('T%d_Freq%d', i+1, j);
    end
end

% 9. 特征融合
fusedFeatures = [baseFeatures, deltaT, tempRate, interactionFeatures,...
                 speedFeatures, tempSpeedInteraction, timeDomainFeatures, freqDomainFeatures];
fprintf('原始特征维度: %d\n', size(fusedFeatures, 2));

% ================== 多工况数据顺序划分策略 ==================
fprintf('采用顺序划分数据集...\n');

% 初始化训练和测试索引
trainIdx = false(size(speed));
testIdx = false(size(speed));

% 存储划分信息
train_test_info = struct();

% 对每个速度按顺序划分（前70%训练，后30%测试）
for s = [6, 8, 10]
    % 找到该速度的所有样本索引
    speed_indices = find(speed == s);
    n_speed = length(speed_indices);
    
    % 计算训练集和测试集数量
    n_train = round(n_speed * 0.7);
    n_test = n_speed - n_train;
    
    % 顺序划分：前70%为训练集，后30%为测试集
    train_indices = speed_indices(1:n_train);
    test_indices = speed_indices(n_train+1:end);
    
    % 标记训练和测试索引
    trainIdx(train_indices) = true;
    testIdx(test_indices) = true;
    
    % 存储划分信息
    train_test_info.(sprintf('speed_%d', s)).total = n_speed;
    train_test_info.(sprintf('speed_%d', s)).train = n_train;
    train_test_info.(sprintf('speed_%d', s)).test = n_test;
    train_test_info.(sprintf('speed_%d', s)).train_ratio = n_train / n_speed;
    train_test_info.(sprintf('speed_%d', s)).test_ratio = n_test / n_speed;
    
    fprintf('%dm/min: 总样本%d, 训练集%d(%.1f%%), 测试集%d(%.1f%%)\n', ...
            s, n_speed, n_train, 100*n_train/n_speed, n_test, 100*n_test/n_speed);
end

% 转换为逻辑索引
trainIdx = logical(trainIdx);
testIdx = logical(testIdx);

% 检查分布
fprintf('\n训练集分布: 6m/min:%d个, 8m/min:%d个, 10m/min:%d个\n',...
    sum(speed(trainIdx)==6),...
    sum(speed(trainIdx)==8),...
    sum(speed(trainIdx)==10));

fprintf('测试集分布: 6m/min:%d个, 8m/min:%d个, 10m/min:%d个\n',...
    sum(speed(testIdx)==6),...
    sum(speed(testIdx)==8),...
    sum(speed(testIdx)==10));

% 输出百分比分布
fprintf('训练集百分比: 6m/min:%.1f%%, 8m/min:%.1f%%, 10m/min:%.1f%%\n',...
    100*mean(speed(trainIdx)==6),...
    100*mean(speed(trainIdx)==8),...
    100*mean(speed(trainIdx)==10));

fprintf('测试集百分比: 6m/min:%.1f%%, 8m/min:%.1f%%, 10m/min:%.1f%%\n',...
    100*mean(speed(testIdx)==6),...
    100*mean(speed(testIdx)==8),...
    100*mean(speed(testIdx)==10));

% 划分数据集
trainInput = fusedFeatures(trainIdx, :);
trainOutput = allOutput(trainIdx);
testInput = fusedFeatures(testIdx, :);
testOutput = allOutput(testIdx);
trainSpeed = speed(trainIdx);
testSpeed = speed(testIdx);

% 保存测试集原始数据，用于后续对比
test_original_data = struct();
for s = [6, 8, 10]
    speed_test_idx = (testSpeed == s);
    if any(speed_test_idx)
        test_original_data.(sprintf('speed_%d', s)).actual = testOutput(speed_test_idx);
        test_original_data.(sprintf('speed_%d', s)).samples = sum(speed_test_idx);
    end
end

% 如果需要，可以将划分信息保存
save('data_split_info.mat', 'train_test_info', 'test_original_data');

% ================== 多工况单独标准化 ==================
fprintf('\n执行多工况单独标准化...\n');

% 初始化标准化后的数组
trainInputNorm = zeros(size(trainInput));
testInputNorm = zeros(size(testInput));
trainOutputNorm = zeros(size(trainOutput));
testOutputNorm = zeros(size(testOutput));

% 存储标准化参数
inputPS = struct();
outputPS = struct();

% 按工况分别标准化
for s = [6, 8, 10]
    % 训练集索引
    trainSpeedIdx = (trainSpeed == s);
    
    if sum(trainSpeedIdx) > 0
        fprintf('  处理 %dm/min: 训练集%d个, ', s, sum(trainSpeedIdx));
        
        % 输入标准化
        [trainNorm, inputPS(s).input] = mapminmax(trainInput(trainSpeedIdx, :)');
        trainInputNorm(trainSpeedIdx, :) = trainNorm';
        
        % 输出标准化
        [outputNorm, outputPS(s).output] = mapminmax(trainOutput(trainSpeedIdx)');
        trainOutputNorm(trainSpeedIdx) = outputNorm';
        
        % 测试集标准化
        testSpeedIdx = (testSpeed == s);
        if sum(testSpeedIdx) > 0
            fprintf('测试集%d个\n', sum(testSpeedIdx));
            testInputNorm(testSpeedIdx, :) = mapminmax('apply', testInput(testSpeedIdx, :)', inputPS(s).input)';
            testOutputNorm(testSpeedIdx) = mapminmax('apply', testOutput(testSpeedIdx)', outputPS(s).output)';
        else
            fprintf('无测试集\n');
        end
    else
        fprintf('  %dm/min: 无训练集数据\n', s);
    end
end

fprintf('\n标准化完成！\n');


% ================== 特征选择 ==================
fprintf('执行特征重要性评估...\n');
% 使用随机森林进行特征重要性评估
mdl = TreeBagger(100, trainInputNorm, trainOutputNorm, 'Method', 'regression',...
                'OOBPredictorImportance', 'on', 'MinLeafSize', 5);

% 获取特征重要性
[importance, idx] = sort(mdl.OOBPermutedPredictorDeltaError, 'descend');
cumImportance = cumsum(importance)/sum(importance);

% 选择累积重要性>95%的特征
retainIdx = find(cumImportance >= 0.95, 1);
selectedFeatures = idx(1:retainIdx);
selectedFeatureNames = featureNames(selectedFeatures);

fprintf('特征选择: 原始特征数 %d => 保留特征数 %d\n',...
    size(trainInputNorm, 2), retainIdx);

% 特征选择验证
fprintf('\n=== 特征选择验证 ===\n');
fprintf('保留特征数: %d\n', retainIdx);
fprintf('累积重要性: %.4f\n', cumImportance(retainIdx));
fprintf('前10个重要特征:\n');
for i = 1:min(10, retainIdx)
    fprintf('  %2d. %s: %.6f\n', i, featureNames{idx(i)}, importance(i));
end

% 更新特征集
trainInputNorm = trainInputNorm(:, selectedFeatures);
testInputNorm = testInputNorm(:, selectedFeatures);

% 特征分析可视化
if featureAnalysis
    figure('Position', [100, 100, 1200, 500], 'Name', '特征分析');
    
    % 特征重要性排序 - 修复版
    subplot(1, 2, 1);
    
    % 只显示选中的特征
    selectedImportance = importance(1:retainIdx);
    selectedIdx = idx(1:retainIdx);
    
    % 按重要性降序排列（最重要的在顶部）
    barh(selectedImportance);
    set(gca, 'YTick', 1:length(selectedIdx), ...
             'YTickLabel', selectedFeatureNames, ...
             'YDir', 'reverse'); % 最重要的特征在顶部
    title('特征重要性排序 (选中的特征)');
    xlabel('重要性得分');
    ylabel('特征名称');
    grid on;
    
    % 累积重要性
    subplot(1, 2, 2);
    plot(1:length(cumImportance), cumImportance, 'LineWidth', 2);
    hold on;
    plot([1, length(cumImportance)], [0.95, 0.95], 'r--', 'LineWidth', 1.5);
    plot(retainIdx, cumImportance(retainIdx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    
    % 标记选定区域
    fill([1, retainIdx, retainIdx, 1], [0, 0, 1, 1], [0.9, 0.95, 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    title('累积特征重要性');
    xlabel('特征数量');
    ylabel('累积重要性');
    xlim([1, length(cumImportance)]);
    ylim([0, 1]);
    grid on;
    legend('累积重要性', '95%阈值', '选定特征数', '选定区域', 'Location', 'southeast');
    
    % 添加文本标注
    text(retainIdx/2, 0.5, sprintf('选定特征\n%d个 (%.1f%%)', retainIdx, cumImportance(retainIdx)*100), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'BackgroundColor', 'white', 'EdgeColor', 'blue', 'FontSize', 10);
    
    saveas(gcf, 'Feature_Importance.png');
end

% ================== 初始化结果存储 ==================
allRMSE = zeros(numRuns, 1);
allMAE = zeros(numRuns, 1);
allMRE = zeros(numRuns, 1);
allMAPE = zeros(numRuns, 1);      % 新增：MAPE
allR2 = zeros(numRuns, 1);        % 新增：决定系数R²
allConvergence = cell(numRuns, 1);
allTime = cell(numRuns, 1);
allGlobalBest = cell(numRuns, 1); 
allTestPredictions = cell(numRuns, 1); 
allFeatureImportance = cell(numRuns, 1); 
bestRunIndex = 0;
bestRMSE = inf;

% 存储最终平均预测结果
avgTestPrediction = [];
avgConvergence = [];

% 主运行循环
for runIdx = 1:numRuns
    fprintf('\n===== 开始运行 %d/%d =====\n', runIdx, numRuns);
    
    % 固定随机种子确保可复现性
    rng(fixedSeed + runIdx);
    
    % 数据标准化 (已提前完成)
    
    % BP神经网络参数
    inputNodes = size(trainInputNorm, 2); 
    fprintf('输入节点数: %d\n', inputNodes);
    
    hiddenNodes = 7;        % 隐含层节点数
    outputNodes = 1;        % 输出层节点数
    learningRate = 0.01;    % 学习速率
    maxIterBP = 500;        % BP 最大迭代次数

    % ILSO 算法参数
    popSize = 80;           % 狮群规模
    maxIterILSO = 200;      % ILSO 最大迭代次数
    beta = 0.2;             % 成年狮子比例
    D = (inputNodes + 1) * hiddenNodes + (hiddenNodes + 1) * outputNodes; % 优化维度
    g_L = -10; g_H = 10;    % 搜索空间边界
    w0 = 0.6; w_min = 0.05; % 调节因子初始参数
    alpha_f = 0.5;          % 母狮局部扰动因子
    alpha_c = 0.8;          % 幼狮局部扰动因子

    % 创新参数 - 混合优化策略
    woaRatio = 0.3;         % 鲸鱼优化参与迭代的比例
    psoRatio = 0.2;         % 粒子群优化参与迭代的比例

    % 创新参数 - 多目标优化
    lambda = 0.5 * (1 - runIdx/numRuns); % 动态调整的复杂度权重

    % 创新参数 - 高斯变异
    gaussSigma = 0.1;       % 高斯变异初始标准差

    %% 2. 初始化狮群位置
    pop = zeros(popSize, D);
    fitness = inf(popSize, 1);
    personalBest = zeros(popSize, D);
    personalBestFitness = inf * ones(popSize, 1);
    globalBest = zeros(1, D);
    globalBestFitness = inf;
    v = zeros(popSize, D); % PSO速度初始化

    % 串行初始化
    for i = 1:popSize
        pop(i, :) = g_L + (g_H - g_L) * rand(1, D); 
        [fitness(i), ~] = calculateFitness(pop(i, :), inputNodes, hiddenNodes, outputNodes, ...
            learningRate, maxIterBP, trainInputNorm', trainOutputNorm', lambda);
        personalBest(i, :) = pop(i, :);
        personalBestFitness(i) = fitness(i);
    end

    % 更新全局最优
    [globalBestFitness, bestIdx] = min(fitness);
    globalBest = pop(bestIdx, :);

    %% 3. ILSO 主迭代过程
    convergenceCurve = zeros(maxIterILSO, 1);
    iterTime = zeros(maxIterILSO, 1);

    % 早停机制参数
    patience = 20;
    bestFitnessHistory = inf(1, patience);

    for iter = 1:maxIterILSO
        startTime = tic;

        % 3.1 计算调节因子 - 非线性sigmoid调节因子
        w = w_min + (w0 - w_min) ./ (1 + exp(-5 * (iter / maxIterILSO - 0.5)));

        % 3.2 划分狮群：狮王、母狮、幼狮
        L = floor(popSize * beta); % 成年狮子数量
        [~, sortIndices] = sort(fitness);
        adultLions = sortIndices(1:L);
        lionKing = adultLions(1); % 狮王
        lionesses = adultLions(2:L); % 母狮
        cubs = sortIndices(L + 1:end); % 幼狮

        % 3.3 混合优化策略
        rnd = rand;
        if rnd < woaRatio
            %% 鲸鱼优化算法(WOA)包围机制
            for i = 1:popSize
                if i ~= lionKing
                    A = 2 * w * rand - w;
                    C = 2 * rand;
                    r = rand;
                    if r < 0.5
                        % 包围猎物
                        D_woa = abs(C * globalBest - pop(i, :));
                        pop(i, :) = globalBest - A * D_woa;
                    else
                        % 搜索猎物
                        randPos = pop(randi(popSize), :);
                        D_woa = abs(C * randPos - pop(i, :));
                        pop(i, :) = randPos - A * D_woa;
                    end
                    % 边界约束
                    pop(i, :) = max(pop(i, :), g_L);
                    pop(i, :) = min(pop(i, :), g_H);
                end
            end
        elseif rnd < woaRatio + psoRatio
            %% 粒子群优化(PSO)速度更新机制
            v_max = 0.2 * (g_H - g_L); % 速度边界
            
            for i = 1:popSize
                w_pso = 0.7 - iter / maxIterILSO * 0.5; % 惯性权重
                c1 = 1.5; c2 = 1.5; % 学习因子
                r1 = rand; r2 = rand;
                % 速度更新
                v(i, :) = w_pso * v(i, :) + c1 * r1 * (personalBest(i, :) - pop(i, :)) + ...
                          c2 * r2 * (globalBest - pop(i, :));
                % 速度边界约束
                v(i, :) = min(max(v(i, :), -v_max), v_max);
                % 位置更新
                pop(i, :) = pop(i, :) + v(i, :);
                % 位置边界约束
                pop(i, :) = max(pop(i, :), g_L);
                pop(i, :) = min(pop(i, :), g_H);
            end
        else
            %% 传统 ILSO 位置更新
            % 3.4.1 狮王位置更新
            r = randn(1, D); % 正态分布随机数
            newKingPos = globalBest + w * (g_H - g_L) * r;
            newKingPos = max(newKingPos, g_L);
            newKingPos = min(newKingPos, g_H);
            pop(lionKing, :) = newKingPos;

            % 3.4.2 母狮位置更新
            [~, bestLionessIdx] = min(personalBestFitness(lionesses));
            bestLionessPos = personalBest(lionesses(bestLionessIdx), :);
            
            for j = 1:length(lionesses)
                i = lionesses(j);
                newLionessPos = (personalBest(i, :) + bestLionessPos) .* (1 + alpha_f * rand) / 2;
                newLionessPos = max(newLionessPos, g_L);
                newLionessPos = min(newLionessPos, g_H);
                pop(i, :) = newLionessPos;
            end

            % 3.4.3 幼狮位置更新
            for j = 1:length(cubs)
                i = cubs(j);
                q = rand;
                if q <= 1/3
                    % 跟随狮王
                    newCubPos = (globalBest + personalBest(i, :)) .* (1 + alpha_c * rand) / 2;
                elseif q <= 2/3
                    % 跟随母狮
                    randLioness = lionesses(randi(length(lionesses)));
                    newCubPos = (personalBest(randLioness, :) + personalBest(i, :)) .* (1 + alpha_c * rand) / 2;
                else
                    % 反向学习
                    g_negative = g_L + g_H - globalBest;
                    newCubPos = (g_negative + personalBest(i, :)) .* (1 + alpha_c * rand) / 2;
                end
                newCubPos = max(newCubPos, g_L);
                newCubPos = min(newCubPos, g_H);
                pop(i, :) = newCubPos;
            end
        end

        % 3.5 高斯变异 - 替换最差个体
        if iter > maxIterILSO * 0.7
            gaussSigma = 0.1 * (1 - iter/maxIterILSO);
            mutated = globalBest + gaussSigma * (g_H - g_L) * randn(1, D);
            mutated = max(min(mutated, g_H), g_L);
            [~, worstIdx] = max(fitness);
            pop(worstIdx, :) = mutated;
        end

        % 3.6 评估新位置
        for i = 1:popSize
            [newFitness, ~] = calculateFitness(pop(i, :), inputNodes, hiddenNodes, outputNodes, ...
                learningRate, maxIterBP, trainInputNorm', trainOutputNorm', lambda);
            
            % 更新个体最优
            if newFitness < personalBestFitness(i)
                personalBest(i, :) = pop(i, :);
                personalBestFitness(i) = newFitness;
            end
            
            fitness(i) = newFitness;
        end
        
        % 更新全局最优
        [currentBestFitness, bestIdx] = min(fitness);
        if currentBestFitness < globalBestFitness
            globalBest = pop(bestIdx, :);
            globalBestFitness = currentBestFitness;
        end

        % 记录收敛曲线和迭代时间
        convergenceCurve(iter) = globalBestFitness;
        iterTime(iter) = toc(startTime);

        % 显示迭代进度
        fprintf('Run %d - ILSO Iteration %d/%d: Fitness = %f, Time = %fs\n', ...
            runIdx, iter, maxIterILSO, globalBestFitness, iterTime(iter));
        
        % 早停机制
        bestFitnessHistory(mod(iter, patience)+1) = globalBestFitness;
        if iter > patience && abs(min(bestFitnessHistory) - max(bestFitnessHistory)) < 1e-6
            fprintf('Early stopping at iteration %d\n', iter);
            convergenceCurve(iter+1:end) = globalBestFitness;
            break;
        end
    end

    % 截断未使用的收敛曲线
    convergenceCurve = convergenceCurve(1:iter);
    iterTime = iterTime(1:iter);

    %% 4. 构建最优 ILSO - BP 模型
    [~, bestBPModel] = calculateFitness(globalBest, inputNodes, hiddenNodes, outputNodes, ...
        learningRate, maxIterBP, trainInputNorm', trainOutputNorm', lambda);

    %% 5. 模型测试与评估
    testPredictionNorm = bestBPModel(testInputNorm'); 
    
    % 反标准化预测结果 (按工况)
    testPrediction = zeros(size(testOutput));
    for s = [6, 8, 10]
        idx = (testSpeed == s);
        if any(idx)
            testPrediction(idx) = mapminmax('reverse', testPredictionNorm(idx)', outputPS(s).output)';
        end
    end
    
    testOutputReal = testOutput; % 实际值不需要反标准化

    % 计算指标
    rmse = sqrt(mean((testPrediction - testOutputReal).^2));
    mae = mean(abs(testPrediction - testOutputReal));
    relativeError = abs((testPrediction - testOutputReal) ./ testOutputReal) * 100;
    meanRelativeError = mean(relativeError);

    % 新增：计算MAPE（平均绝对百分比误差）
    mape = mean(abs((testPrediction - testOutputReal) ./ testOutputReal)) * 100;

    % 新增：计算决定系数R²
    ss_res = sum((testPrediction - testOutputReal).^2);
    ss_tot = sum((testOutputReal - mean(testOutputReal)).^2);
    r2 = 1 - (ss_res / ss_tot);
    
    % 存储结果
    allRMSE(runIdx) = rmse;
    allMAE(runIdx) = mae;
    allMRE(runIdx) = meanRelativeError;
    allMAPE(runIdx) = mape;    % 新增
    allR2(runIdx) = r2;        % 新增
    allConvergence{runIdx} = convergenceCurve;
    allTime{runIdx} = iterTime;
    allGlobalBest{runIdx} = globalBest;
    allTestPredictions{runIdx} = testPrediction;
    
    % 计算特征重要性
    weights = bestBPModel.IW{1};
    featureImportance = mean(abs(weights), 1)';
    allFeatureImportance{runIdx} = featureImportance;
    
    % 记录最佳运行
    if rmse < bestRMSE
        bestRMSE = rmse;
        bestRunIndex = runIdx;
    end
    
    % 累积预测结果用于平均
    if isempty(avgTestPrediction)
        avgTestPrediction = testPrediction;
    else
        avgTestPrediction = avgTestPrediction + testPrediction;
    end
    
    fprintf('Run %d 测试结果:\n', runIdx);
    fprintf('RMSE: %f μm\n', rmse);
    fprintf('MAE: %f μm\n', mae);
    fprintf('平均相对误差: %f%%\n', meanRelativeError);
    fprintf('MAPE: %f%%\n', mape);          % 新增
    fprintf('决定系数R²: %f\n\n', r2);      % 新增
    
    %% 6. 单次运行结果可视化
    if showIndividualPlots
        figure('Position', [100, 100, 1200, 800], 'Name', sprintf('Run %d Results', runIdx));

        % 绘制收敛曲线
        subplot(2, 2, 1);
        plot(1:length(convergenceCurve), convergenceCurve, 'LineWidth', 2);
        title(sprintf('Run %d: ILSO 收敛曲线', runIdx));
        xlabel('迭代次数');
        ylabel('适应度值(RMSE)');
        grid on;

        % 绘制特征重要性
        subplot(2, 2, 2);
        bar(featureImportance);
        set(gca, 'XTick', 1:length(selectedFeatureNames), 'XTickLabel', selectedFeatureNames, 'XTickLabelRotation', 45);
        title(sprintf('Run %d: 特征重要性', runIdx));
        xlabel('特征');
        ylabel('重要性得分');
        grid on;

        % 绘制预测结果与实际值对比
        subplot(2, 2, 3);
        plot(1:length(testOutputReal), testOutputReal, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(1:length(testPrediction), testPrediction, 'r--', 'LineWidth', 1.5);
        title(sprintf('Run %d: 预测 vs 实际值', runIdx));
        xlabel('样本序号');
        ylabel('热误差(μm)');
        legend('实际值', '预测值');
        grid on;

        % 绘制误差分布
        subplot(2, 2, 4);
        error = testPrediction - testOutputReal;
        histogram(error, 10, 'Normalization', 'probability');
        title(sprintf('Run %d: 预测误差分布', runIdx));
        xlabel('预测误差(μm)');
        ylabel('概率密度');
        grid on;
    end
end

% 平均预测结果
avgTestPrediction = avgTestPrediction / numRuns;

% 计算平均收敛曲线
maxLen = max(cellfun(@length, allConvergence));
avgConvergence = zeros(maxLen, 1);
for i = 1:maxLen
    validRuns = 0;
    sumVal = 0;
    for j = 1:numRuns
        if i <= length(allConvergence{j})
            sumVal = sumVal + allConvergence{j}(i);
            validRuns = validRuns + 1;
        end
    end
    avgConvergence(i) = sumVal / validRuns;
end

% 计算统计指标
meanRMSE = mean(allRMSE);
stdRMSE = std(allRMSE);
meanMAE = mean(allMAE);
stdMAE = std(allMAE);
meanMRE = mean(allMRE);
stdMRE = std(allMRE);
meanMAPE = mean(allMAPE);    % 新增
stdMAPE = std(allMAPE);      % 新增
meanR2 = mean(allR2);        % 新增
stdR2 = std(allR2);          % 新增

% 最佳运行结果
bestTestOutput = testOutputReal; % 所有运行测试集相同

fprintf('\n===== %d 次运行平均结果 =====\n', numRuns);
fprintf('平均 RMSE: %.4f ± %.4f μm\n', meanRMSE, stdRMSE);
fprintf('平均 MAE: %.4f ± %.4f μm\n', meanMAE, stdMAE);
fprintf('平均相对误差: %.2f%% ± %.2f%%\n', meanMRE, stdMRE);
fprintf('平均 MAPE: %.2f%% ± %.2f%%\n', meanMAPE, stdMAPE);      % 新增
fprintf('平均决定系数R²: %.4f ± %.4f\n', meanR2, stdR2);        % 新增
fprintf('最佳运行: #%d (RMSE = %.4f μm)\n', bestRunIndex, bestRMSE);

%% 7. 绘制综合结果
figure('Position', [100, 100, 1400, 900], 'Name', '综合结果');

% 平均收敛曲线
subplot(2, 3, 1);
hold on;
for i = 1:numRuns
    plot(1:length(allConvergence{i}), allConvergence{i}, 'Color', [0.7, 0.7, 0.7]);
end
plot(1:length(avgConvergence), avgConvergence, 'r-', 'LineWidth', 2);
title('平均收敛曲线');
xlabel('迭代次数');
ylabel('适应度值(RMSE)');
legend('单次运行', '平均');
grid on;

% 指标分布箱线图
subplot(2, 3, 2);
allMetrics = [allRMSE; allMAE; allMRE; allMAPE; allR2];  % 修改：包含所有指标
groupLabels = [repmat({'RMSE'}, numRuns, 1); 
               repmat({'MAE'}, numRuns, 1);
               repmat({'MRE'}, numRuns, 1);
               repmat({'MAPE'}, numRuns, 1);             % 新增
               repmat({'R²'}, numRuns, 1)];              % 新增
boxplot(allMetrics, groupLabels);
title('指标分布');
ylabel('值');
grid on;

% 预测结果与实际值对比 (最佳运行)
subplot(2, 3, 3);
plot(1:length(bestTestOutput), bestTestOutput, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:length(allTestPredictions{bestRunIndex}), allTestPredictions{bestRunIndex}, 'r--', 'LineWidth', 1.5);
title(sprintf('最佳运行 #%d: 预测 vs 实际值', bestRunIndex));
xlabel('样本序号');
ylabel('热误差(μm)');
legend('实际值', '预测值');
grid on;

% 平均预测结果与实际值对比
subplot(2, 3, 4);
plot(1:length(bestTestOutput), bestTestOutput, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:length(avgTestPrediction), avgTestPrediction, 'm--', 'LineWidth', 1.5);
title('平均预测 vs 实际值');
xlabel('样本序号');
ylabel('热误差(μm)');
legend('实际值', '平均预测');
grid on;

% 误差分布直方图 (所有运行)
subplot(2, 3, 5);
allErrors = [];
for i = 1:numRuns
    currentErrors = allTestPredictions{i} - bestTestOutput;
    allErrors = [allErrors; currentErrors];
end
histogram(allErrors, 20, 'Normalization', 'probability');
title('综合误差分布');
xlabel('预测误差(μm)');
ylabel('概率密度');
grid on;

% 特征重要性平均
subplot(2, 3, 6);
avgFeatureImportance = zeros(size(allFeatureImportance{1}));
for i = 1:numRuns
    avgFeatureImportance = avgFeatureImportance + allFeatureImportance{i};
end
avgFeatureImportance = avgFeatureImportance / numRuns;
bar(avgFeatureImportance);
set(gca, 'XTick', 1:length(selectedFeatureNames), 'XTickLabel', selectedFeatureNames, 'XTickLabelRotation', 45);
title('平均特征重要性');
xlabel('特征');
ylabel('重要性得分');
grid on;

%% 8. 多工况结果分析
% 按工况分离测试结果
speedTest = testSpeed; 

% 各工况性能评估
for s = [6, 8, 10]
    idx = (speedTest == s);
    if any(idx)
        % 使用平均预测结果
        pred = avgTestPrediction(idx); 
        actual = testOutputReal(idx);
        
        rmse = sqrt(mean((pred - actual).^2));
        mae = mean(abs(pred - actual));
        mre = mean(abs(pred - actual)./abs(actual)) * 100;
        mape = mean(abs((pred - actual) ./ actual)) * 100;                    % 新增
        ss_res = sum((pred - actual).^2);                                     % 新增
        ss_tot = sum((actual - mean(actual)).^2);                             % 新增
        r2 = 1 - (ss_res / ss_tot);                                           % 新增
        
        fprintf('\n工况 %dm/min 性能:\n', s);
        fprintf('RMSE: %.4f μm\n', rmse);
        fprintf('MAE: %.4f μm\n', mae);
        fprintf('平均相对误差: %.2f%%\n', mre);
        fprintf('MAPE: %.2f%%\n', mape);                                      % 新增
        fprintf('决定系数R²: %.4f\n', r2);                                    % 新增
        
        % 工况专属可视化
        figure('Name', sprintf('%dm/min 预测对比', s));
        plot(actual, 'b-o', 'LineWidth', 1.5, 'DisplayName', '实际值');
        hold on;
        plot(pred, 'r--s', 'LineWidth', 1.5, 'DisplayName', '预测值');
        title(sprintf('%dm/min 热误差预测', s));
        xlabel('样本序号');
        ylabel('热误差(μm)');
        legend show;
        grid on;
        saveas(gcf, sprintf('Speed_%dm_Prediction.png', s));
    end
end

% 速度-误差关系分析 (新增过渡区标记)
figure('Name', '速度-误差关系');
absErrors = abs(avgTestPrediction - testOutputReal);
speedLabels = arrayfun(@(x) sprintf('%dm/min', x), speedTest, 'UniformOutput', false);
boxplot(absErrors, speedLabels);
hold on;

% 标记速度过渡区
transition_points = find(diff(speedTest) ~= 0);
for tp = transition_points'
    plot(tp, max(absErrors)*1.1, 'r*', 'MarkerSize', 10);
end
xlabel('进给速度');
ylabel('绝对误差 (μm)');
title('不同速度下预测误差分布');
grid on;
saveas(gcf, 'Error_by_Speed.png');

%% 保存预测结果用于对比分析
fprintf('\n=== 保存预测结果用于算法对比 ===\n');

% 创建结果结构体
comparisonResults = struct();

% 按工况分离结果
speedTest = testSpeed;
for s = [6, 8, 10]
    idx = (speedTest == s);
    if any(idx)
        % 使用平均预测结果
        pred = avgTestPrediction(idx); 
        actual = testOutputReal(idx);
        
        % 存储结果
        comparisonResults.(sprintf('speed_%d', s)).ILSO_Hybrid_actual = actual;
        comparisonResults.(sprintf('speed_%d', s)).ILSO_Hybrid_predictions = pred;
        
        fprintf('已保存 %dm/min 的ILSO-BP混合优化预测结果\n', s);
    end
end

% 保存到文件
save('ILSO_Hybrid_Comparison_Results.mat', 'comparisonResults');
fprintf('ILSO-BP混合优化预测结果已保存到 ILSO_Hybrid_Comparison_Results.mat\n');                                                                                                                        
%% 修复后的适应度计算函数
function [fitness, bpModel] = calculateFitness(params, inputNodes, hiddenNodes, outputNodes, ...
    learningRate, maxIter, trainInput, trainOutput, lambda)
    
    % 创建新网络对象
    bpModel = feedforwardnet(hiddenNodes);
    
    % ================== 关键修复：显式设置网络结构 ==================
    % 配置网络参数
    bpModel.trainParam.lr = learningRate;
    bpModel.trainParam.epochs = maxIter;
    bpModel.trainParam.goal = 1e-4;
    bpModel.trainParam.showWindow = false;
    
    % 显式设置网络层大小
    bpModel.inputs{1}.size = inputNodes;   % 设置输入层大小
    bpModel.layers{1}.size = hiddenNodes;  % 设置隐含层大小
    bpModel.layers{2}.size = outputNodes;  % 设置输出层大小
    
    % 手动设置权重和阈值
    w1_size = inputNodes * hiddenNodes;
    b1_size = hiddenNodes;
    w2_size = hiddenNodes * outputNodes;
    b2_size = outputNodes;
    
    % 从参数向量中提取权重和阈值
    w1 = reshape(params(1:w1_size), inputNodes, hiddenNodes);
    b1 = reshape(params(w1_size+1:w1_size+b1_size), 1, hiddenNodes);
    w2 = reshape(params(w1_size+b1_size+1:w1_size+b1_size+w2_size), hiddenNodes, outputNodes);
    b2 = reshape(params(end-b2_size+1:end), 1, outputNodes);
    
    % 设置网络权重和阈值
    paramVector = [w1(:); b1(:); w2(:); b2(:)];
    bpModel = setwb(bpModel, paramVector);
    
    % 验证网络结构
    if bpModel.inputs{1}.size ~= inputNodes
        error('网络输入层大小(%d)与期望(%d)不匹配', bpModel.inputs{1}.size, inputNodes);
    end
    
    % 训练网络
    [bpModel, ~] = train(bpModel, trainInput, trainOutput);

    % 计算训练集上的 RMSE
    trainPrediction = bpModel(trainInput);
    rmse = sqrt(mean((trainPrediction - trainOutput).^2));

    % 使用L2正则化
    modelComplexity = sum(params.^2); % L2范数
    fitness = rmse + lambda * modelComplexity;
end
