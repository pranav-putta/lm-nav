from lmnav.common.registry import registry

@registry.register_fn('filter_methods.dtg')
def dtg_filter_fn(config, episode):
    return len(episode) < 500 and episode[-1]['info']['distance_to_goal'] <= config.dtg_threshold
 
