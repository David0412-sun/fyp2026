function Selftest1_compact(fileName)
if nargin<1, fileName='test0929.wav'; end

% ==== 参数 ====
N=512; c=343; Dia=0.08; r=Dia/2; M=8;
rhoL=0.01; rhoH=0.5; Ksmooth=5;
cohTh=1.7; vadTh=6; noiseA=0.98; srpA=0.85;
fmin=300; fmax=3400; nR=81; 
rhoGrid=linspace(rhoL,rhoH,nR); %根据距离上限和距离下限生成均分的网格
dTheta=-6:2:6;

% ==== 文件/阵列 ====
info=audioinfo(fileName); fs=info.SampleRate; T=info.TotalSamples/fs;
afr=dsp.AudioFileReader('Filename',fileName,'SamplesPerFrame',N);
ang=(0:M-1)*45; mic=[r*cosd(ang(:)) r*sind(ang(:))]; Pairs=nchoosek(1:M,2);

% ==== 图 ====
f=figure('Name','GCC角度 + SRP距离','NumberTitle','off'); tl=tiledlayout(f,2,1,'Padding','compact','TileSpacing','compact');
ax1=nexttile(tl,1); hold(ax1,'on'); h1=plot(ax1,nan,nan,'b--'); h2=plot(ax1,nan,nan,'r-'); xlim(ax1,[0 T]); ylim(ax1,[-180 180]); grid(ax1,'on'); title(ax1,'Angle vs Time');
ax2=nexttile(tl,2); hold(ax2,'on'); h3=plot(ax2,nan,nan,'g--'); h4=plot(ax2,nan,nan,'m-'); xlim(ax2,[0 T]); ylim(ax2,[0 1]); grid(ax2,'on'); title(ax2,'Distance vs Time');
axp=polaraxes('Parent',f,'Position',[0.65,0.55,0.30,0.38]); hold(axp,'on'); axp.ThetaZeroLocation='right'; axp.ThetaDir='counterclockwise'; axp.ThetaTick=0:30:330; axp.RLim=[0 r*1.2]; title(axp,'Angle (fixed radius)');
for m=1:M, polarplot(axp,deg2rad(ang(m)),r,'k^','MarkerFaceColor','y','MarkerSize',8); end
hl=polarplot(axp,nan,nan,'r-','LineWidth',1.5); hm=polarplot(axp,nan,nan,'ro','MarkerSize',8,'LineWidth',2);

% ==== 缓存 ====
t=[]; Adeg=[]; Rho=[]; treal=0; prevRho=NaN; tauMax=Dia/c; srpAcc=zeros(1,nR); noiseRMS=[]; win=0.5*(1-cos(2*pi*(0:N-1)'/N));

% ==== 主循环 ====
while ~isDone(afr)
    x=afr(); xw=x.*repmat(win,1,M); treal=treal+N/fs; %加hann窗实现帧头帧尾平滑过渡
    
    frmRMS=median(sqrt(mean(xw.^2,1))); %计算加窗后信号的均方根
    if isempty(noiseRMS), noiseRMS=frmRMS; end %第一帧将frmRMS作为背景噪声基准
    snrDb=20*log10((frmRMS+1e-12)/(noiseRMS+1e-12)); %计算信噪比
    if snrDb<vadTh, noiseRMS=noiseA*noiseRMS+(1-noiseA)*frmRMS; end %指数移动平均法

    [tdoa, wcoh]=gcc_sub(xw, Pairs, N, fs, tauMax); %计算时延
    [udir,adeg]=dir_est(mic, Pairs, c, tdoa); %最小二乘计算角度
    if sum((pairA(mic,Pairs)*udir).* (c*tdoa))<0, udir=-udir; adeg=wrap180(adeg+180); end %若u方向雨实际点积方向相反，旋转180度

    Xi=fft(xw,N); good = (snrDb>=vadTh) && (median(wcoh)>=cohTh);%判断足够的信噪比和相干度
    if ~good
        rho = tern(isnan(prevRho), 0.20, prevRho);%若不靠谱：判断上一个距离是否为nan，若是为默认值0.20m，若否将沿用上一个距离
    else
        bestIdx=1; %最佳距离网格索引，初设为第一个
        bestU=udir; %最佳方向向量，初设为上一步计算的主方向
        bestA=adeg; %最佳角度，初设为上一步的角度估计
        for th=adeg+dTheta %在角度估计+-6°做2°为步进的微调生成
            u=[cosd(th) sind(th)]; %七个候选方向
            sc=scores_pair(Xi,fs,N,c,mic,u,rhoGrid,fmin,fmax); %计算得分
            tmp=srpA*srpAcc+(1-srpA)*sc; %EMA把历史得分与当前得分融合
            [~,idx]=max(tmp);%找出最大值的索引
            if tmp(idx) > srpAcc(bestIdx), bestIdx=idx; bestU=u; bestA=th; end%更新最佳距离索引，对应的方向bestU=u和角度bestA=th
        end
        sc = scores_pair(Xi,fs,N,c,mic,bestU,rhoGrid,fmin,fmax); 
        srpAcc=srpA*srpAcc+(1-srpA)*sc; [~,bestIdx]=max(srpAcc);
        rho=rhoGrid(bestIdx); udir=bestU; adeg=bestA; %使用刚才更新的最佳方向bestU再精算一次，并更新对应距离，角度，方向
    end

    t(end+1)=treal; Adeg(end+1)=adeg; Rho(end+1)=rho; %将当前时间戳，角度，距离进行存储
    A1=movmean(Adeg,Ksmooth); R1=movmean(Rho,Ksmooth); %滑动平均

    set(hl,'ThetaData',[deg2rad(adeg) deg2rad(adeg)],'RData',[0 r]); set(hm,'ThetaData',deg2rad(adeg),'RData',r);
    set(h1,'XData',t,'YData',Adeg); set(h2,'XData',t,'YData',A1);
    set(h3,'XData',t,'YData',Rho);  set(h4,'XData',t,'YData',R1);
    drawnow limitrate; prevRho=rho; pause(N/fs);
end
release(afr); end

% ==== 子函数 ====
function [tdoa,w]=gcc_sub(xw,Pairs,N,fs,tauMax)
P=size(Pairs,1); tdoa=zeros(P,1); w=ones(P,1);
for p=1:P
    i=Pairs(p,1); j=Pairs(p,2);
    Xi=fft(xw(:,i),N); Xj=fft(xw(:,j),N); R=Xi.*conj(Xj); R=R./(abs(R)+eps);
    cc=ifft(R,N,'symmetric'); cc=fftshift(cc);
    [pk,idx]=max(abs(cc)); d=(parab(cc,idx)); lag=(idx-1)-N/2 + d; %对互相关序列做抛物线插值后求时延
    med=median(abs(cc))+eps; w(p)=max(0.1,min(5.0,pk/med)); %找互相关序列的中位数med，将pk/med的结果限制在[0.1,5.0]的范围内，防止极大极小值
    t=lag/fs; tdoa(p)=max(-tauMax,min(t,tauMax));%计算时间差t=lag/fs，t不能超过阵列直径所限制的实验极值[-Dia/c,Dia/c]
end
end

function d=parab(y,i)
N=numel(y); iL=mod(i-2,N)+1; i0=mod(i-1,N)+1; iR=mod(i,N)+1;
d=(y(iL)-y(iR))/(2*(y(iL)-2*y(i0)+y(iR)+eps));
end

function A=pairA(mic,Pairs)
P=size(Pairs,1); A=zeros(P,2);
for p=1:P, A(p,:)=mic(Pairs(p,2),:)-mic(Pairs(p,1),:); end
end

function [u,adeg]=dir_est(mic,Pairs,c,tdoa)
A=pairA(mic,Pairs); b=c*tdoa; uEst=A\b; u=uEst/(norm(uEst)+1e-12); adeg=atan2d(u(2),u(1));%最小二乘法计算声源方位向量u，二维反正切计算角度
end

function sc=scores_pair(Xi,fs,N,c,mic,u,rhos,fmin,fmax)
M=size(mic,1); fk=(0:N-1)'; fk=fk*(fs/N); band=(fk>=fmin & fk<=fmax); fk=fk(band);%获得所有频率点后截取人声区间
X=Xi(band,:); X=X./(abs(X)+eps);%对应频带的频谱并作归一化
Pairs=nchoosek(1:M,2); P=size(Pairs,1); %获取麦克风对矩阵
R=complex(zeros(numel(fk),P)); %互谱矩阵，存放第p对麦在第f个频率点的互谱
for p=1:P, i=Pairs(p,1); 
    j=Pairs(p,2); 
    R(:,p)=X(:,i).*conj(X(:,j)); %逐对计算互谱
end
sc=zeros(1,numel(rhos)); u=reshape(u,1,[]);%sc会保存“假设声源在方向 u 且距离 rhos(k) 处时”的 SRP-PHAT 得分；将列向量u转为行向量
for k=1:numel(rhos)
    s=rhos(k)*u; %假设声源在方向 u 且距离 rhos(k) 处
    D=mic-repmat(s,M,1); %计算每只麦克风相对声源的向量差；mic是M*2的阵列坐标，对应每个麦克风的坐标；repmat(s,M,1) 把 s 重复到 M*2，相减后 D(i,:) 就是第 i 支麦到假设声源的向量
    dist=sqrt(sum(D.^2,2)); %距离：距离向量差的模
    tau=dist/c; %计算时延
    tauij=tau(Pairs(:,2))-tau(Pairs(:,1)); %如果声源在s处，信号应该先后到达1，2麦的时间差
    Ftau=repmat(fk,1,numel(tauij)).*repmat((tauij.'),numel(fk),1);%相位补偿指数部分，频率乘以时延
    W=exp(-1i*2*pi*Ftau); %相位补偿：预测时延下频域多出的部分
    sc(k)=real(sum(sum(R.*W))); %理论相位差与实际相位差相乘，越精确相位就越一致，相乘结果越大
end
end

function y=tern(c,a,b), if c, y=a; else, y=b; end, end
function a=wrap180(a), a=mod(a+180,360)-180; end
