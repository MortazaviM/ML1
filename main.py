import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt("D:\python\web_traffic.tsv", delimiter="\t")

x=data[:,0]
y=data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]
plt.scatter(x,y)
plt.title("Web Traffic")
plt.xlabel("time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i' %w for w in range(10)])

plt.autoscale(tight=True)
plt.grid()



'''error function'''
def error(f,x,y):
    return sp.sum((f(x)-y)**2)

fp1,res,rank,sv,rcond=sp.polyfit(x,y,1,full=True)
print(fp1)

f1=sp.poly1d(fp1)
print(f1)

print(error(f1, x, y))

fx=sp.linspace(0, x[-1], 1000)
plt.plot(fx,f1(fx), linewidth=4, color='green')


fp2=sp.polyfit(x,y,2)
f2=sp.poly1d(fp2)
plt.plot(fx,f2(fx), linewidth=4, color='red')


fp3=sp.polyfit(x,y,3)
f3=sp.poly1d(fp3)
plt.plot(fx,f3(fx), linewidth=4, color='yellow')

fp10=sp.polyfit(x,y,10)
f10=sp.poly1d(fp10)
plt.plot(fx,f10(fx), linewidth=4, color='black')



fp100=sp.polyfit(x,y,100)
f100=sp.poly1d(fp100)
plt.plot(fx,f100(fx), linewidth=4, color='violet')
plt.show()

print(f2)




























