from concurrent import futures
import logging
import json
import threading
from datetime import datetime
import pandas as pd

import grpc
from grpc import StatusCode
import grpc_pb2
import grpc_pb2_grpc

from classes.trainer import Trainer, predict

# Define a shared variable to hold the object returned from the background task
shared_lock = threading.Lock()

server_address = '83.212.75.52:50051'

# Define the background task function
def background_task(config_dict, target):
    """
    This function will be run in a separate thread.
    """
    print("Starting background task for target: " + target)
    # create trainer object which start the training
    trainer = Trainer(config_dict, target)
    trainer.start()
    print("Finished background task for target: " + target)
    # Return the trainer object
    return trainer

# RouteGuideServicer provides an implementation of the methods of the RouteGuide service.
class RouteGuideServicer(grpc_pb2_grpc.RouteGuideServicer):
    def __init__(self):
        self.job_id = ''
        self.results = {}
        self.status = {}
        self.trainers = {}

    def StartTraining(self, request, context):
        # Clean the self variables
        self.job_id = ''
        self.results = {}
        self.status = {}
        self.trainers = {}

        # Read the config from the request
        self.job_id = request.id
        self.results[self.job_id] = {}
        self.status['id'] = self.job_id
        config_dict = json.loads(request.config)

        # for each target column, create a status with waiting
        for target in config_dict['targetColumn']:
          self.status[target] = 'waiting'

        # Start the background task in a separate thread
        thread = threading.Thread(target=self.background_task_wrapper, args=(request, config_dict))
        thread.start()

        return grpc_pb2.Status(id=self.job_id, status='started')
    
    def background_task_wrapper(self, request, config_dict):
        job_id = request.id
         # create trainer object for each target column in the config
        for target in config_dict['targetColumn']:
          self.results[job_id][target] = {}
          # set the status to processing
          self.status[target] = 'processing'
          # Run the background task and store the returned object in the shared variable
          self.trainers[target] = background_task(config_dict, target)
          # When the background task is done, update the shared variable

          with shared_lock:
              # set the status to done
              self.status[target] = 'done'

              # save the results of the training from the thread to the results dictionary
              self.results[job_id][target] = self.trainers[target].get_results()

        # Results are ready and can be sent to the client
        # TODO: send the results to orchestrator/controller

    def GetProgress(self, request, context):
        """
        Get progress for a specific job
        Return: The job id and the status of each target column
        """
        job_id = request.id

        if job_id in self.results:
            # Create a Struct message
            data = {} 
            # Add the status of each target column to the struct
            for target in self.status:
              data[target] = self.status[target]

            # delete the id from the struct
            del data['id']
            return grpc_pb2.Progress(id=job_id, data=data)
        else:
            # return empty response
            context.abort(StatusCode.INVALID_ARGUMENT, "Not a valid job id")
    
    def GetSpecificTargetResults(self, request, context):
      """
      Get the results for a specific target column
      Return: The predictions and evaluation metrics for each model
      """
      target = request.name
      job_id = request.id

      if job_id in self.results:
        if target in self.results[job_id]:
          if self.status[target] == 'done':
            data = {}
            for model in self.results[job_id][target]:
              # assign the prediction results for each model
              data[model] = grpc_pb2.Predictions(predictions=self.results[job_id][target][model]['predictions'],
                                                 evaluation={
                                                  'MSE': self.results[job_id][target][model]['evaluation']['MSE'],
                                                  'MAE': self.results[job_id][target][model]['evaluation']['MAE'],
                                                  'MAPE': self.results[job_id][target][model]['evaluation']['MAPE'],
                                                  'RMSE': self.results[job_id][target][model]['evaluation']['RMSE']
                                                 })
          
            return grpc_pb2.Results(target=target,metrics=data)
          else:
            # return empty response
            context.abort(StatusCode.INVALID_ARGUMENT, "Task has not finished yet")
      else:
          # return empty response
          context.abort(StatusCode.INVALID_ARGUMENT, "Not a valid job id or target column")
          
    def GetAllTargetsResults(self, request, context):
      '''
      Get the results for all target columns
      Return: The predictions and evaluation metrics for each model for each target column
      '''
      job_id = request.id
      # create an empty response - array of dictionaries where each dictionary is a target column
      all_results = grpc_pb2.AllResults()

      if job_id in self.results:
        data = {}
        for target in self.results[job_id]:
          data[target] = {}
          if self.status[target] == 'done':
            for model in self.results[job_id][target]:
              data[target][model] = grpc_pb2.Predictions(predictions=self.results[job_id][target][model]['predictions'],
                                                        evaluation={
                                                        'MSE': self.results[job_id][target][model]['evaluation']['MSE'],
                                                        'MAE': self.results[job_id][target][model]['evaluation']['MAE'],
                                                        'MAPE': self.results[job_id][target][model]['evaluation']['MAPE'],
                                                        'RMSE': self.results[job_id][target][model]['evaluation']['RMSE']
                                                        })

          # add the target column and its results to the response
          all_results.results.append(grpc_pb2.Results(target=target, metrics=data[target]))
        return all_results
      else:
          # return empty response
          context.abort(StatusCode.INVALID_ARGUMENT, "Not a valid job id")
         
    def GetInference(self, request, context):
      # get the timestamp from the request and convert it to a datetime object
      date = datetime.fromtimestamp(request.timestamp)
      model_name = request.model_name

      # Convert date to dataframe and set it as the index
      date = pd.DataFrame({'timestamp': [date]})
      date['timestamp'] = pd.to_datetime(date['timestamp'])
      date = date.set_index('timestamp')
      
      # get the predictions for the given timestamp and assign to Any type
      y_pred = predict(date, model_name)
      return grpc_pb2.Inference(predictions=y_pred)
    
    def SaveModel(self, request, context):
      # get the information
      model_type = request.model_type
      model_name = request.model_name
      target = request.target

      # verify that target exist in trainers
      if target in self.trainers:
        status = self.trainers[target].save_model(model_type, model_name, target)
        return grpc_pb2.Status(id=self.job_id, status=status)
      else:
        # return empty response
        context.abort(StatusCode.INVALID_ARGUMENT, "Task has not finished yet or not exists")


def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  grpc_pb2_grpc.add_RouteGuideServicer_to_server(
      RouteGuideServicer(), server)
  server.add_insecure_port(server_address)
  # print the url to the console so we can connect
  print(f"Starting server. Listening on {server_address}")
  server.start()
  server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
    # thread = threading.Thread(target=serve)
    # thread.start()