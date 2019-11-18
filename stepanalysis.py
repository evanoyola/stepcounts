import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as dates
import datetime as dt

### open file

data=pd.read_csv('stepsdata.csv')

### basic data description

print(data.shape)
print(data.columns)
print(data.dtypes)
print(data.head(8))

### transform timestamp and make steps integers

date = data['Start']
datetime = pd.to_datetime(date)

steps = data['Steps (count)']
steps = steps.astype(int)

datesteps = pd.concat([datetime, steps], axis=1)
datesteps.columns = ['Date','Steps']

print(datesteps.head(8))

### first analysis

ave=steps.mean()
print('Averange steps', ave)


### add month, weekday

datestep=datesteps.set_index('Date',inplace=True)
datesteps['Month'] = datesteps.index.month_name()
datesteps['Weekday'] = datesteps.index.weekday_name

#print(datesteps.shape)
#print(datesteps.head(8))
#print(datesteps.dtypes)


### eliminate "low" days

datestepsclean=datesteps[datesteps['Steps']>500]

#print(datestepsclean.shape)

meanclean=datestepsclean.mean()
print('clean mean', np.int(meanclean[0]))

### highest date

max = datesteps.max()
maxsteps=max['Steps']
datemax=datesteps.loc[datesteps['Steps'] == maxsteps].index[0]
tmax = datemax.strftime('%B %d, %Y')
print('Historical maximum',maxsteps, 'steps on', tmax)


###Time series handling

#Year

seventeen=datesteps.loc['2017']
eightteen=datesteps.loc['2018']
nineteen=datesteps.loc['2019']

#Weekday

mo=datesteps[datesteps.Weekday.astype(str).str.contains('Monday')]
tu=datesteps[datesteps.Weekday.astype(str).str.contains('Tuesday')]
we=datesteps[datesteps.Weekday.astype(str).str.contains('Wednesday')]
th=datesteps[datesteps.Weekday.astype(str).str.contains('Thursday')]
fr=datesteps[datesteps.Weekday.astype(str).str.contains('Friday')]
sa=datesteps[datesteps.Weekday.astype(str).str.contains('Saturday')]
su=datesteps[datesteps.Weekday.astype(str).str.contains('Sunday')]

weekmean=[np.int(mo.mean()),np.int(tu.mean()),np.int(we.mean()),np.int(th.mean()),np.int(fr.mean()),np.int(sa.mean()),np.int(su.mean())]
weeksigma=[np.int(mo.std()),np.int(tu.std()),np.int(we.std()),np.int(th.std()),np.int(fr.std()),np.int(sa.std()),np.int(su.std())]

#print(weekmean)
#print(weeksigma)

#Season

spri=datesteps[(datesteps.Month.astype(str).str.contains('March'))|(datesteps.Month.astype(str).str.contains('April'))]
summ=datesteps[(datesteps.Month.astype(str).str.contains('May'))|(datesteps.Month.astype(str).str.contains('June'))|(datesteps.Month.astype(str).str.contains('July'))|(datesteps.Month.astype(str).str.contains('August'))|(datesteps.Month.astype(str).str.contains('September'))]
fall=datesteps[(datesteps.Month.astype(str).str.contains('October'))|(datesteps.Month.astype(str).str.contains('November'))]
wint=datesteps[(datesteps.Month.astype(str).str.contains('December'))|(datesteps.Month.astype(str).str.contains('January'))|(datesteps.Month.astype(str).str.contains('February'))]

seasonmean=[np.int(spri.mean()),np.int(summ.mean()),np.int(fall.mean()),np.int(wint.mean())]
seasonsigma=[np.int(spri.std()),np.int(summ.std()),np.int(fall.std()),np.int(wint.std())]

#print(seasonmean)
#print(seasonsigma)

###Reset index

datesteps=datesteps.reset_index()
seventeen=seventeen.reset_index()
eightteen=eightteen.reset_index()
nineteen=nineteen.reset_index()


###Convert to numpy

npdatesteps = datesteps.to_numpy() 
npseventeen = seventeen.to_numpy() 
npeightteen = eightteen.to_numpy() 
npnineteen = nineteen.to_numpy()


#For plotting

tsteps=np.array(npdatesteps[:,1], dtype=float)
dseven=npseventeen[:,0]
sseven=np.array(npseventeen[:,1], dtype=float)
deight=npeightteen[:,0]
seight=np.array(npeightteen[:,1], dtype=float)
dnine=npnineteen[:,0]
snine=np.array(npnineteen[:,1], dtype=float)


###plt.plot(data)

#plt.tight_layout()
csfont = {'fontname':'Times New Roman'}

fig = plt.figure(constrained_layout=False)
grid = plt.GridSpec(4, 5, wspace=0.4, hspace=0.3)
fig.suptitle("Step Analysis",fontsize=15,**csfont)


##print mean, max

ax3=plt.subplot(grid[0:2, 0:3])

ax3.spines["top"].set_visible(False)    
ax3.spines["bottom"].set_visible(False)    
ax3.spines["right"].set_visible(False)    
ax3.spines["left"].set_visible(False)

ax3.set_xlim([0,10])
ax3.set_ylim([0,10])
ax3.xaxis.set_ticks_position('none')
ax3.yaxis.set_ticks_position('none')

ax3.set_xticks([])
ax3.set_yticks([])

bbox_props = dict(boxstyle="circle,pad=0.7", fc="yellow", alpha=0.4, ec="black", lw=2)
t = ax3.text(0.8, 5.9, "                ", ha="center", va="center",size=10,bbox=bbox_props)
ax3.annotate('Average', xy=(0,0), xytext=(-0.2,6.7),fontsize=13,**csfont)
ax3.annotate('Steps', xy=(0,0), xytext=(0.1,5.6),fontsize=13,**csfont)
ax3.annotate(np.int(ave), xy=(0,0), xytext=(0,4),fontsize=16, weight='bold',**csfont)

bbox_props = dict(boxstyle="circle,pad=0.3", fc="green", alpha=0.3, ec="black", lw=2)
t = ax3.text(4.8, 3, "                        ", ha="center", va="center",size=10,bbox=bbox_props)
ax3.annotate("Maximum", xy=(0,0), xytext=(3.6,4.5),fontsize=12,**csfont)
ax3.annotate("steps", xy=(0,0), xytext=(4.2,3.8),fontsize=12,**csfont)
ax3.annotate(maxsteps, xy=(0,0), xytext=(3.8,2.5),fontsize=15,weight='bold',**csfont)
ax3.annotate("on", xy=(0,0), xytext=(4.5,1.7),fontsize=9,**csfont)
ax3.annotate(tmax, xy=(0,0), xytext=(3.4,1),fontsize=9,**csfont)



bbox_props = dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3, ec="black", lw=2)
t = ax3.text(4, 8, "Female", ha="center", va="center",size=8,bbox=bbox_props,**csfont)
bbox_props = dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3, ec="black", lw=2)
t = ax3.text(7, 6.5, "Age 44", ha="center", va="center",size=8,bbox=bbox_props,**csfont)
bbox_props = dict(boxstyle="round,pad=0.3", fc="red", alpha=0.3, ec="black", lw=2)
t = ax3.text(1, 2, "Austin", ha="center", va="center",size=8,bbox=bbox_props,**csfont)


##Bar plots

ax4=plt.subplot(grid[0, 3:5])

ax4.spines["top"].set_visible(False)    
ax4.spines["bottom"].set_visible(True)    
ax4.spines["right"].set_visible(False)    
ax4.spines["left"].set_visible(True)

width=0.5
weeklabels=['Mo','Tu','We','Th','Fr','Sa','Su']

ax4.bar(weeklabels, weekmean, width, color='gray', edgecolor=['red', 'red', 'red', 'red', 'red','green', 'green'], alpha=0.8)

for tick in ax4.get_xticklabels():
    tick.set_fontname(**csfont)
for tick in ax4.get_yticklabels():
    tick.set_fontname(**csfont)

#ax4.set_ylabel(**csfont)
ax4.set_title('Average Steps',**csfont)

###

ax5=plt.subplot(grid[1, 3:5])

ax5.spines["top"].set_visible(False)    
ax5.spines["bottom"].set_visible(True)    
ax5.spines["right"].set_visible(False)    
ax5.spines["left"].set_visible(True)

width=0.7
seasonlabels=['Spring','Summer','Fall','Winter']

ax5.barh(seasonlabels, seasonmean, width, color=['green', 'red', 'orange', 'blue'], edgecolor='black', xerr=seasonsigma, alpha=0.6)

for tick in ax5.get_xticklabels():
    tick.set_fontname(**csfont)
for tick in ax5.get_yticklabels():
    tick.set_fontname(**csfont)


##Step timeseries

ax1=plt.subplot(grid[2:4, 0:3])

ax1.spines["top"].set_visible(False)    
ax1.spines["bottom"].set_visible(True)    
ax1.spines["right"].set_visible(False)    
ax1.spines["left"].set_visible(True)


ax1.set_xlim([dt.date(2017,1,1), dt.date(2019,9,1)])
ax1.set_ylim([0, 32000])
ax1.set_xlabel('Time', fontsize=10,**csfont)
ax1.set_ylabel('Steps', fontsize=10,**csfont)


#x_datesn = [dt.date(2017,1,1),dt.date(2017,4,1),dt.date(2017,7,1), dt.date(2017,10,1),dt.date(2018,1,1),dt.date(2018,4,1),dt.date(2018,7,1), dt.date(2018,10,1),dt.date(2019,1,1),dt.date(2019,4,1),dt.date(2019,7,1)]
x_datesn = []
ax1.set_xticklabels(labels=x_datesn, rotation=45,**csfont)
ax1.xaxis.set_ticks_position('none')
#ax1.xaxis.set_major_locator(plt.MaxNLocator(3))

#myFmt = x_datesn.DateFormatter('%y%m')
#ax.xaxis.set_major_formatter(myFmt)


y_stepsn = [0,5000,10000,15000,20000,25000]
ax1.set_yticklabels(labels=y_stepsn, rotation=45,**csfont)
ax1.set_yticks(y_stepsn)

plt.plot(dseven,sseven, color='green', alpha=0.5) 
plt.plot(deight,seight, color='red', alpha=0.5) 
plt.plot(dnine,snine, color='blue', alpha=0.5) 


ax1.annotate('2017', xy=(dt.date(2017,5,1), 27000), xytext=(dt.date(2017,5,1), 27000),**csfont)
ax1.annotate('2018', xy=(dt.date(2018,5,1), 27000), xytext=(dt.date(2018,5,1), 27000),**csfont)
ax1.annotate('2019', xy=(dt.date(2019,3,1), 27000), xytext=(dt.date(2019,3,1), 27000),**csfont)


##Step histogram


ax2=plt.subplot(grid[2:4, 3:5])

ax2.spines["top"].set_visible(False)    
ax2.spines["bottom"].set_visible(True)    
ax2.spines["right"].set_visible(False)    
ax2.spines["left"].set_visible(False) 

ax2.set_xlim([-1000, 28000])
#ax2.set_xlabel('Steps', fontsize=10,**csfont)


plt.hist(tsteps, 40, density=False, facecolor='yellow',edgecolor='white', alpha=0.7)
plt.hist(sseven, 47, density=False, facecolor='green',edgecolor='white', alpha=0.35)
plt.hist(seight, 43, density=False, facecolor='red',edgecolor='white', alpha=0.45)
plt.hist(snine, 38, density=False, facecolor='blue',edgecolor='white', alpha=0.45)

x_stepsn = [0,5000,10000,15000,20000,25000]
ax2.set_xticklabels(labels=x_stepsn, rotation=45,**csfont)
ax2.yaxis.set_ticks_position('none')
ax2.set_xticks(x_stepsn)

y_datesn = []
ax2.set_yticklabels(labels=y_datesn, rotation=45,**csfont)

ax2.annotate('Step Histogram', xy=(13500, 80), xytext=(9000, 85),fontsize=12,**csfont)
bbox_props = dict(boxstyle="circle,pad=0.3", fc="yellow", alpha=0.6, ec="black", lw=1)
t = ax2.text(16000, 67, "   ", ha="center", va="center",size=5,bbox=bbox_props,**csfont)
ax2.annotate('total', xy=(18000, 70), xytext=(18000, 65),**csfont)
bbox_props = dict(boxstyle="circle,pad=0.3", fc="green", alpha=0.5, ec="black", lw=1)
t = ax2.text(16000, 57, "   ", ha="center", va="center",size=5,bbox=bbox_props,**csfont)
ax2.annotate('2017', xy=(18000, 60), xytext=(18000, 55),**csfont)
bbox_props = dict(boxstyle="circle,pad=0.3", fc="red", alpha=0.5, ec="black", lw=1)
t = ax2.text(16000, 47, "   ", ha="center", va="center",size=5,bbox=bbox_props,**csfont)
ax2.annotate('2018', xy=(18000, 50), xytext=(18000, 45),**csfont)
bbox_props = dict(boxstyle="circle,pad=0.3", fc="blue", alpha=0.5, ec="black", lw=1)
t = ax2.text(16000, 37, "   ", ha="center", va="center",size=5,bbox=bbox_props,**csfont)
ax2.annotate('2019', xy=(18000, 40), xytext=(18000, 35),**csfont)


plt.show()

fig.savefig("steps.pdf")
fig.savefig("steps.jpg")


