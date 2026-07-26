import numpy as np
from src.env.entities import User, MECServer, AuxVehicle
from src.env.channel import ChannelModel

def main():
    # 1. Khởi tạo
    mec = MECServer(id=0, position=np.array([0.0, 0.0]), cpu_frequency=1e9) # 1GHz
    aux = AuxVehicle(id=1, position=np.array([50.0, 10.0]), cpu_frequency=0.5e9)
    user = User(id=2, position=np.array([-250.0, 5.0]), cpu_frequency=0.5e9)
    channel = ChannelModel()
    

    # 2. Sinh Task
    user.generate_task()
    print(f"User Task: {user.current_task.size / 8 / 1024:.2f} KB")

    # 3. Kiểm tra vùng phủ sóng
    print(f"Distance to MEC: {mec.get_distance(user):.2f} m")
    print(f"Is Covered by MEC? {mec.is_covered(user)}")

    # 4. Tính tốc độ truyền
    rate_mec = channel.get_uplink_rate_mec(mec.get_distance(user), user.tx_power)
    print(f"Uplink Rate to MEC: {rate_mec / 1e6:.2f} Mbps")

    rate_v2v = channel.get_uplink_rate_v2v(aux.get_distance(user), user.tx_power)
    print(f"Uplink Rate to AuxVehicle: {rate_v2v / 1e6:.2f} Mbps")

    # 5. Di chuyển User
    user.step_mobility(dt=1.0)
    print(f"New User Position: {user.position}")

if __name__ == "__main__":
    main()