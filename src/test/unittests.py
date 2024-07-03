def test_ik(physics, target_pos=None, target_quat=None): 
    qpos = physics.data.qpos
    site_xpos = physics.data.site_xpos[-1,:]
    
    print("Initial qpos: ", qpos)
    print("Initial xpos: ", site_xpos)
    if target_pos is None:
        target_pos = site_xpos
    
    ik_result = ik.qpos_from_site_pose(physics, 'attachment_site', target_pos, target_quat=None)
    return ik_result, ik_result.successs