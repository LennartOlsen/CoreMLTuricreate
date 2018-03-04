//
//  ViewController.swift
//  CoreMLTuricreate
//
//  Created by Lennart Olsen on 04/03/2018.
//  Copyright © 2018 lennartolsen.net. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    let p = Predictor()

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        print(p.PerformPrediction())
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewDidAppear(_ animated: Bool) {
        print(p.PerformPrediction())
    }


}

