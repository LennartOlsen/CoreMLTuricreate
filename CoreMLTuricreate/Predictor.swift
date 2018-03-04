//
//  Predictor.swift
//  SensorTagRaiderMacOS
//
//  Created by Lennart Olsen on 19/02/2018.
//  Copyright © 2018 lennartolsen.net. All rights reserved.
//

import Foundation
import CoreML

class Predictor {
    
    struct ModelConstants {
        static let numOfFeatures = 9
        static let predictionWindowSize = 4
        static let sensorsUpdateInterval = 1.0 / 4.0
        static let hiddenInLength = 200
        static let hiddenCellInLength = 200
    }
    
    var predictionWindowDataArray : MLMultiArray?
    var lastHiddenCellOutput: MLMultiArray?
    var lastHiddenOutput: MLMultiArray?
    
    let activityClassificationModel = timeFramedActivityClassifier()
    
    init(){
        predictionWindowDataArray = try? MLMultiArray(shape : [1,ModelConstants.predictionWindowSize,ModelConstants.numOfFeatures] as [NSNumber], dataType : MLMultiArrayDataType.double)
        lastHiddenOutput = try? MLMultiArray(shape:[ModelConstants.hiddenInLength as NSNumber], dataType: MLMultiArrayDataType.double)
        lastHiddenCellOutput = try? MLMultiArray(shape:[ModelConstants.hiddenCellInLength as NSNumber], dataType: MLMultiArrayDataType.double)
    }
    
    func PerformPrediction() -> String? {
        // Add the current accelerometer reading to the data array
        guard let dataArray = predictionWindowDataArray else { return "Error" }
        
        /**
         Dummy data
         **/
        dataArray[[0 , 0 , 0] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 1] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 2] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 3] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 4] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 5] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 6] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 7] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 0 , 8] as [NSNumber]] = Double(0.1220703125) as NSNumber
        
        
        dataArray[[0 , 1 , 0] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 1] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 2] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 3] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 4] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 5] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 6] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 7] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 1 , 8] as [NSNumber]] = Double(0.1220703125) as NSNumber
        
        dataArray[[0 , 2 , 0] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 1] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 2] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 3] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 4] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 5] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 6] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 7] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 2 , 8] as [NSNumber]] = Double(0.1220703125) as NSNumber
        
        
        dataArray[[0 , 3 , 0] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 1] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 2] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 3] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 4] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 5] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 6] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 7] as [NSNumber]] = Double(0.1220703125) as NSNumber
        dataArray[[0 , 3 , 8] as [NSNumber]] = Double(0.1220703125) as NSNumber
        
        guard let prediction = try? activityClassificationModel.prediction(features: dataArray,
                                                                          hiddenIn: lastHiddenOutput,
                                                                          cellIn: lastHiddenCellOutput) else {
                                                                            return "N/A"
        }
        
        // Update the state vectors
        lastHiddenOutput = prediction.hiddenOut
        lastHiddenCellOutput = prediction.cellOut
        
        return prediction.type
    }
}
