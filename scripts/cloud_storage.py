from azure.storage.blob import BlockBlobService

class CloudStorage(object):
    def __init__(self):
        pass

    def upload():
        block_blob_service = BlockBlobService(account_name='amsword',
                account_key='aNkZljf+FCvdHMWRTJxsCo+8beMMHzVQy/2rH87sjDuoXYf5PzrSj+Z5Vkm35/PovDfXe/Ev6B1ymKe4nJl50w==')
        container_name = 'dataset'
        blob_name = 'abc'
        file_name = ''
        block_blob_service.create_blob_from_path(container_name,
                blob_name, file_name)

if __name__ == '__main__':
    pass

