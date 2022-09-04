from pathlib import Path
data_dir = Path(__file__).parent.parent / "data"

from vidstab import VidStab
import matplotlib.pyplot as plt

stabilizer = VidStab()
video_in = str(data_dir / "example_raw.avi")
video_out = str(data_dir / "example_stable.avi")
print(video_in)
stabilizer.stabilize(input_path=video_in, output_path=video_out)

stabilizer.plot_trajectory()
plt.show()

stabilizer.plot_transforms()
plt.show()