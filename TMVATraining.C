
void TMVATraining(){
TMVA::Tools::Instance();
auto outputFile=TFile::Open("TMVA.root","RECREATE");

TMVA::Factory factory("tmva1",outputFile, "!V:ROC:!Silent:Color:!DrawProgressBar:AnalysisType=Classification");

TFile *inputS =TFile::Open("Higgs_signal.root");
TFile *inputB =TFile::Open("Higgs_background.root");

//get TTree objects from input file
TTree *signalTree = (TTree*)inputS->Get("Events");
TTree *backgroundTree = (TTree*)inputB->Get("Events");

//Declare the dataloader class
TMVA::DataLoader * loader = new TMVA::DataLoader("dataset");
// global event weights per tree (see below for setting event-wise weights)
Double_t signalWeight     = 1.0;
Double_t backgroundWeight = 1.0;

loader->AddSignalTree    (signalTree, signalWeight);  
loader->AddBackgroundTree(backgroundTree,backgroundWeight);




signalTree->Print();
// Add the variables  

loader->AddVariable("Muon_pt_1");
loader->AddVariable("Muon_pt_2");
loader->AddVariable("Electron_pt_1");
loader->AddVariable("Electron_pt_2");
loader->AddVariable("dPhim1e1");
loader->AddVariable("dPhim1e2");
loader->AddVariable("dPhim2e1");
loader->AddVariable("dPhim2e2");
loader->AddVariable("dPhim1m2");
loader->AddVariable("dPhie1e2");
loader->AddVariable("Electron_eta_1");
loader->AddVariable("Electron_eta_2");
loader->AddVariable("Muon_eta_1");
loader->AddVariable("Muon_eta_2");


// cuts at 10<JetPt<100
TCut mycuts = ""; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
TCut mycutb = ""; // for example: TCut mycutb = "abs(var1)<0.5";
//TCut mycuts = "JetsPt>80 && JetsPt<100 "; // for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
//TCut mycutb = "JetsPt>80 && JetsPt<100 "; // for example: TCut mycutb = "abs(var1)<0.5";



//loader->PrepareTrainingAndTestTree( mycuts, mycutb,
  //                                     "SplitMode=random!V" );// split input data into training and test data.
     loader->PrepareTrainingAndTestTree( mycuts,mycutb, "SplitMode=random:!V" );                                   
                                       
  
                   
factory.BookMethod(loader,TMVA::Types::kBDT, "BDTG",
                   "!V:NTrees=500:MinNodeSize=2%:MaxDepth=4:BoostType=Grad:Shrinkage=0.05:UseBaggedBoost:"
                   "BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20:CreateMVAPdfs=True:DoBoostMonitor=True:VarTransform=N");
//factory.BookMethod(loader,TMVA::Types::kBDT, "BDTG 2000",
//                   "!V:NTrees=2000:MinNodeSize=.25%:MaxDepth=4:BoostType=Grad:Shrinkage=0.01:UseBaggedBoost:"
//                   "BaggedSampleFraction=0.5:SeparationType=GiniIndex");

                   


//MVA method:MLP 
//factory.BookMethod(loader, TMVA::Types::kMLP, "MLP","!H:!V:HiddenLayers=2");
//MVA method:DNN                      
bool useDLCPU = false;
   bool useDLGPU = false;
#ifdef R__HAS_TMVACPU
   useDLCPU = true;
#endif
#ifdef R__HAS_TMVAGPU
   useDLGPU = true;
#endif

if (useDLCPU || useDLGPU) {  
      // Define DNN layout
      TString inputLayoutString = "InputLayout=1|1|14"; 
      TString batchLayoutString= "BatchLayout=1|256|14";
      TString layoutString ("Layout=DENSE|100|RELU,DENSE|100|RELU,DENSE|100|RELU,DENSE|1|LINEAR");
      // Define Training strategies 
      // one can catenate several training strategies 
      TString training1("LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
                        "ConvergenceSteps=10,BatchSize=256,TestRepetitions=1,"
                        "MaxEpochs=30,Regularization=NONE,"
                        "Optimizer=ADAM");
         TString training2("LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
                             "ConvergenceSteps=10,BatchSize=256,TestRepetitions=1,"
                             "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
                             "Optimizer=ADAGRAD,DropConfig=0.0+0.0+0.0+0.");
  
      TString trainingStrategyString ("TrainingStrategy=");
      trainingStrategyString += training1;  //+ "|" //+ training2;

      // General Options.

      TString dnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:"
                          "WeightInitialization=XAVIER");
      dnnOptions.Append (":"); dnnOptions.Append (inputLayoutString);
      dnnOptions.Append (":"); dnnOptions.Append (batchLayoutString);
      dnnOptions.Append (":"); dnnOptions.Append (layoutString);
      dnnOptions.Append (":"); dnnOptions.Append (trainingStrategyString);

      if (useDLGPU)
         dnnOptions += ":Architecture=GPU";
      else
         dnnOptions += ":Architecture=CPU";
      factory.BookMethod(loader, TMVA::Types::kDL, "DL_CPU", dnnOptions);
  }
  

      factory.TrainAllMethods();
      


  // ## Test  all methods
 //Now we test and evaluate all methods using the test data set


   factory.TestAllMethods();   

   factory.EvaluateAllMethods();
  //factory.OptimizeAllMethods("ROCIntegral","Minuit2");
   
/// after we get the ROC curve and we display 

   auto c1 = factory.GetROCCurve(loader);
   c1->Draw();

/// at the end we close the output file which contains the evaluation result of all methods and it can be used by TMVAGUI
/// to display additional plots
   
   outputFile->Close();


}

// After finishing training you can get the plots using tmva gui.
//$root -l 
//$TMVA::TMVAGUI("TMVA1.root")



































 

