import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description='Plot the open-loop torque replay comparison')
p.add_argument('directory',type=Path)
a=p.parse_args()
d=a.directory
with (d/'replay.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
if not rows:
    raise ValueError('No replay samples')
def col(name): return np.array([float(r[name]) for r in rows])
t=col('sample_time_s'); tc=col('control_time_s'); u=col('replayed_torque_Nm')
q0=col('baseline_position_rad'); q1=col('damped_position_rad')
v0=col('baseline_velocity_rad_s'); v1=col('damped_velocity_rad_s')
if not np.isfinite(np.c_[t,tc,u,q0,q1,v0,v1]).all(): raise ValueError('Non-finite data')
summary=json.loads((d/'summary.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(2,2,figsize=(12,8),sharex=True,layout='constrained')
ax[0,0].plot(tc,u,color='#555555',label='Same input for both runs')
ax[0,0].set(title='Replayed joint1 torque',ylabel='Applied torque (N·m)'); ax[0,0].legend()
ax[0,1].plot(t,q0,color='#1768ac',label=f"Original b={summary['baseline_damping']:g}")
ax[0,1].plot(t,q1,color='#d45b12',label=f"Changed b={summary['changed_damping']:g}")
ax[0,1].set(title='Joint1 position',ylabel='Position (rad)'); ax[0,1].legend()
ax[1,0].plot(t,v0,color='#1768ac',label='Original'); ax[1,0].plot(t,v1,color='#d45b12',label='Changed')
ax[1,0].set(title='Joint1 velocity',ylabel='Velocity (rad/s)',xlabel='Time (s)'); ax[1,0].legend()
ax[1,1].plot(t,q1-q0,color='#d45b12')
ax[1,1].set(title='Position: changed minus original',ylabel='Position difference (rad)',xlabel='Time (s)')
for a in ax.flat:
    a.axhline(0,color='#888',linewidth=.7); a.axvline(.4,color='#999',linestyle=':',linewidth=1); a.grid(alpha=.25)
fig.suptitle('Panda joint1: same open-loop torque, different damping',fontsize=14,fontweight='bold')
fig.savefig(d/'replay_comparison.png',dpi=170)
print(f'Saved {d / "replay_comparison.png"}')
