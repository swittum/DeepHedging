import matplotlib.pyplot as plt
from handler import Handler

handler = Handler('./config.yaml')
history, results = handler.run()

# TODO: Reorder config.yaml (features array)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(history)
ax2.hist(results, bins=200)
plt.savefig('results.png', dpi=300)