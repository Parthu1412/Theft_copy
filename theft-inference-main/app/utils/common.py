import pickle
import redis


def singleton(cls):
    """Decorator to ensure a class has only one instance."""
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return get_instance


@singleton
class CacheHelper:
    def __init__(self):
        self.redis_cache = redis.StrictRedis(
            host="localhost", port=6379, db=0, socket_timeout=1
        )
        print("REDIS CACHE UP!")

    def get_redis_pipeline(self):
        return self.redis_cache.pipeline()

    def set_json(self, dict_obj):
        """Store a dictionary in Redis as a serialized object."""
        try:
            k, v = list(dict_obj.items())[0]
            v = pickle.dumps(v)
            return self.redis_cache.set(k, v)
        except redis.ConnectionError:
            return None

    def get_json(self, key):
        """Retrieve and deserialize a stored object from Redis, then delete the key."""
        try:
            temp = self.redis_cache.get(key)
            if temp:
                # Delete the key after retrieving the value
                self.redis_cache.delete(key)
                return pickle.loads(temp)
            return None
        except redis.ConnectionError:
            return None
