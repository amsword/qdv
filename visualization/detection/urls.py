from django.conf.urls import url
from . import views

app_name = 'detection'

urlpatterns = [
    url(r'^view_image/(.*)/(.*)/(.*)/([0-9]+)$', 
        views.view_image, 
        name='view_image'),
    url(r'taxonomy_validation_result/$',
        views.validate_taxonomy,
        name='taxonomy_validation_result'),
    url(r'^view_image/$', 
        views.view_image2, 
        name='view_image2'),
    url(r'^taxonomy_verification/$',
        views.input_taxonomy,
        name='taxonomy_verification'),
    url(r'^view_tree/$', 
        views.view_tree, 
        name='view_tree'),
    url(r'^check_mapping/$', 
        views.check_mapping, 
        name='check_mapping'),
    # if no parameter is given -> show all the exps
    # if full_expid is given -> show all the prediction results, i.e. *.predict
    # file
    # if full_expid is given and file name of predict file is given -> show the
    # prediction results. 
    # old scenario: full_expid is not specified. 
    url(r'^view_model/$', 
        views.view_model,
        name='view_model'),
    url(r'^media/(.*)$',
        views.download_file,
        name='return_file'),
]

