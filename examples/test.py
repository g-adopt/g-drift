import gdrift
import matplotlib.pyplot as plt

a = gdrift.load_dataset("SLB_21_pyroliteCFMAS")
a["bulk_mod"]

fig = plt.figure(num=1)
ax = fig.add_subplot(111)
ax.plot(a["rho"][100,:])
fig.show()