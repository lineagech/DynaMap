#include <llvm/Pass.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h> // for DI...
#include <llvm/IR/CFG.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>

#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#include <llvm/Support/raw_ostream.h>

#include <llvm/CodeGen/RegAllocRegistry.h>

#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/ValueTracking.h>

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include "inc.def"
#include "./thread_idx_analysis.hpp"

//#define QUANT_LINE_SIZE 32
#define INSERT_BEFORE 1

using namespace llvm;
using namespace std;

static string blockIdx_y_str;
static string threadIdx_y_str;
static string blockDim_y_str;

namespace {
    typedef enum {
        LOAD_INST = 0,
        STORE_INST
    } MEM_INST_TYPE;

    struct MemAccInst : public FunctionPass {
        /* ID is used for LLVM to identify pass */
        bool isInit;
        string cuda_inst_func_name;
        string malloc_managed_inst_func_name; 
        Function* cuda_inst_func;
        Function* channel_end_func;
        DataLayout* module_data_layout;
        
        unordered_set<string> kernel_set;
        unordered_set<string> global_ptr_set;
        unordered_set<string> global_ptr_addr_set;
        unordered_set<string> processed_func_set;

        unordered_set<Value*> global_ptr_value_set;
        
        /* Statistics */
        int num_of_malloc_managed;
        int num_of_inserted_load;
        int num_of_inserted_store;

        static char ID;
        /* functions names or built-in names */
        static string cudaMallocManagedName;
        static string statTableFuncName;
        static string devRecvWithLineFuncName;
        static string update_PST_1D_Func_Name;
        string update_PST_2D_Block_X_Func_Name;
        string update_PST_2D_Block_Y_Func_Name;
        string devRecv_Func_Name;
        string push_pst_2d_x_func_name;
        string push_pst_2d_y_func_name;
        string inst_stat_table_alloc_func_name; 
        static string update_PST_1D_Thread_Func_Name;

        static string threadIdx_x_str;
        static string blockIdx_x_str;
        static string blockDim_x_str;
            
        ThreadIdxAnalysis* tid_analysis;

        MemAccInst() : FunctionPass(ID) {
            isInit = false;
            num_of_malloc_managed = 0;
            num_of_inserted_load = 0;
            num_of_inserted_store = 0;
            tid_analysis = new ThreadIdxAnalysis();
        }

        void getAnalysisUsage(AnalysisUsage &AU) const override {
            //AU.addRequired<BasicAAWrapperPass>();
            AU.addRequired<AAResultsWrapperPass>();
        }
       
        void insertChannelInitBlock(Function* func)
        {
            Module* module = func->getParent();
            LLVMContext& context = func->getContext();
            //BasicBlock* block = BasicBlock::Create(context, "channel_dev_init", func, &(*func->begin()));
            //IRBuilder<> builder(block);
            Instruction* first_inst = &(*(inst_begin(func)));
            IRBuilder<> builder(first_inst);

            /* look for host_init */
            Function* host_init_func = NULL;
            for (auto mod_func = module->begin(); mod_func != module->end(); mod_func++) {
                if (mod_func->getName().str().find("channel_host_init") != string::npos) {
                    host_init_func = &(*mod_func);
                    break;
                }
            }

            assert(host_init_func);
            
            IntegerType* int_type = IntegerType::get(context, 32);
            ConstantInt* const_num_malloc = ConstantInt::get(int_type, num_of_malloc_managed, false);
            Value* args[] = {const_num_malloc};
            
            builder.CreateCall(host_init_func, args);        
            //builder.CreateRetVoid();
            errs() << "Insert Channel Init Function at Host side" << host_init_func->getName() 
                << ", with malloc num " << num_of_malloc_managed << "!\n\n\n";
        }
        
        void countManagedMalloc(Module* module)
        {
            for (auto func = module->begin(); func != module->end(); func++) {
                //if (func->getName().str() == "main") {
                if (func->getName().str().find("mem_acc_stat_table_alloc") != string::npos) 
                    continue;
                if (func->getName().str().find("channel_host_init") == string::npos
                    && func->getName().str().find(cudaMallocManagedName) == string::npos) {
                    for (auto inst = inst_begin(&*func); inst != inst_end(&*func); inst++) {
                        if (auto call_inst = dyn_cast<CallInst>(&(*inst))) {
                            auto called_func = call_inst->getCalledValue();
                            if (called_func->getName().str().find(cudaMallocManagedName) != string::npos) {
                                errs() << __func__ << ": " << called_func->getName() <<"\n";
                                num_of_malloc_managed++;
                            }
                        }
                        else if (auto invoke_inst = dyn_cast<InvokeInst>(&*inst)) {
                            auto called_value = invoke_inst->getCalledOperand();
                            if (called_value->getName().str().find(cudaMallocManagedName) != string::npos) {
                                errs() << "Invoke --- Called Value --- " << called_value->getName() << "\n";
                                called_value->dump();
                                num_of_malloc_managed++;
                            }
                        }
                    }
                    
                }
            }
        }

        bool collectKernels(Module* module) 
        {
            bool kernel_found = false;
            for (auto named_meta = module->named_metadata_begin(); 
                     named_meta != module->named_metadata_end(); 
                     named_meta++) {
                for (auto op = named_meta->op_begin(); op != named_meta->op_end(); op++) {
                    MDNode* node = (*op);
                    Metadata* prev_meta = NULL;
                    for (auto sub_op = node->op_begin(); sub_op != node->op_end(); sub_op++) {
                        Metadata* meta = sub_op->get();
                        if (!meta) continue;
                        MDString* meta_str = dyn_cast<MDString>(meta);

                        if (meta_str && meta_str->getString().str() == "kernel") {
                            kernel_found = true;
                            string md_str;
                            raw_string_ostream md_str_stream(md_str);
                            prev_meta->print(md_str_stream);

                            string kernel_name = md_str.substr(md_str.find("@")+1);
                            
                            if (kernel_name.find("psu_cuc1057_channel_device_init") == string::npos) {
                                errs() << "\t Found a kernel function: "
                                    << kernel_name << "\n";
                                kernel_set.insert(kernel_name);
                            }
                        }
                        prev_meta = meta;
                    }
                }
            }
            return kernel_found;
        }
        
        void collectInstrumentFunc(Module* module)
        {
            for (auto func = module->begin(); func != module->end(); func++) {
                if (func->getName().str().find("update_PST_2d_block_x") != string::npos) {
                    update_PST_2D_Block_X_Func_Name = func->getName().str();
                }
                if (func->getName().str().find("update_PST_2d_block_y") != string::npos) {
                    update_PST_2D_Block_Y_Func_Name = func->getName().str();
                }
                else if (func->getName().str().find("update_PST_1d") != string::npos) {
                    update_PST_1D_Func_Name = func->getName().str();
                }
                else if (func->getName().str().find("deviceRecv") != string::npos && 
                         func->getName().str().find("WithLine") == string::npos) {
                    devRecv_Func_Name = func->getName().str();
                }
                else if (func->getName().str().find("push_pst_2d_x") != string::npos) {
                    push_pst_2d_x_func_name = func->getName().str(); 
                }
                else if (func->getName().str().find("push_pst_2d_y") != string::npos) {
                    push_pst_2d_y_func_name = func->getName().str(); 
                }
                else if (func->getName().str().find("mem_acc_stat_table_alloc") != string::npos) {
                    inst_stat_table_alloc_func_name = func->getName().str();
                }
            }
        }

        string getInstrumentFunc(Module* module)
        {
            string instr_func_name;
            for (auto func = module->begin(); func != module->end(); func++) {
                if (func->getName().str().find("deviceRecv") != string::npos) {
                    instr_func_name = func->getName().str();
                    errs() << "================= Found instrument code: " << func->getName() << " ==================\n\n";
                    break;
                }
            }
            return instr_func_name;
        }
        
        Function* getChannelEndFunc(Module* module) 
        {
            /* Insert channel_end function */
            for (auto mod_func = module->begin(); mod_func != module->end(); mod_func++) {
                if (mod_func->getName().str().find("channel_host_end") != string::npos) {
                    return (&(*mod_func));
                }
            }
        }
        
        Function* getStatTableFunc(Module* module) 
        {
            /* Insert channel_end function */
            for (auto mod_func = module->begin(); mod_func != module->end(); mod_func++) {
                if (mod_func->getName().str().find(statTableFuncName) != string::npos) {
                    return (&(*mod_func));
                }
            }
        }

        void insertFuncBeforeRet(Function* func, Function* target_func) 
        {
            /* insert inst before every return instruction */
            for (auto func_inst = inst_begin(func), func_inst_end = inst_end(func); 
                 func_inst != func_inst_end; func_inst++) {
                if ( auto ret_inst = dyn_cast<ReturnInst>(&(*func_inst)) ) {
                    //ret_inst->getParent()->getInstList().insert(func_inst, inst);
                    CallInst* new_call_inst = CallInst::Create(dyn_cast<Value>(target_func));
                    Instruction* new_inst = dyn_cast<Instruction>(new_call_inst);
                    assert(new_inst);
                    //func_inst->getParent()->getInstList().insert(&(*func_inst), new_inst);
                    new_inst->insertBefore(&(*func_inst));
                    //dyn_cast<Instruction>(new_call_inst)->insertBefore(&(*func_inst));
                    errs() << "\n\n ======= Insert channel end before =======";
                    func_inst->dump();
                    new_inst->dump();
                    errs() << "==================\n\n\n";
                }
            }
        }
    
        void insertNotProcessedFuncs()
        {
            processed_func_set.insert(update_PST_1D_Func_Name);
            processed_func_set.insert(update_PST_2D_Block_X_Func_Name);
            processed_func_set.insert(devRecv_Func_Name);
            processed_func_set.insert(push_pst_2d_x_func_name);
            processed_func_set.insert(push_pst_2d_y_func_name);
            processed_func_set.insert("_ZN10ChannelDev24get_PST_index_given_addrEm");
            processed_func_set.insert("_Z13update_PST_1dPvii");
            processed_func_set.insert("_Z17channel_host_initi");
            processed_func_set.insert(inst_stat_table_alloc_func_name);
        }

        /* overloads an abstract virtual method inherited from FunctionPass */
        virtual bool runOnFunction(Function& fun) 
        {
            LLVMContext& context = fun.getContext();
            Module *module = fun.getParent();
            //errs() << "Func: " << fun.getName() << ", Module: " << module->getName() << "\n";
           
            if (!isInit) {
                //errs() << "Init...... \n";
                module_data_layout = new DataLayout(module);
                collectKernels(module);       
                countManagedMalloc(module);
                cuda_inst_func_name = getInstrumentFunc(module);
                collectInstrumentFunc(module);
                
                for (auto mod_func = module->begin(); mod_func != module->end(); mod_func++) {
                    if (mod_func->getName().str() == "main") {
                        /* Handle main function */
                        insertChannelInitBlock(&(*mod_func));
                        
                        /* Insert channel end function */
                        //Function* channel_host_end_func = getChannelEndFunc(module);
                        Function* target_func = getChannelEndFunc(module);
                        insertFuncBeforeRet(&(*mod_func), target_func);
                    }
                }
                insertNotProcessedFuncs();
                isInit = true;
            }
            
            if (processed_func_set.find(fun.getName().str()) != processed_func_set.end()) {
                return true;
            }

            /* only process kernel functions and their subordinates */
            if (kernel_set.find(fun.getName().str()) != kernel_set.end()) { 
                tid_analysis->buildLoadStoreUse(&fun);
                identifyGlobalAddr(&fun, context, module);
                processInstrument(&fun, context, module);
                errs() << "# of Inserted Load " << num_of_inserted_load << "\n";
                errs() << "# of Inserted Store " << num_of_inserted_store << "\n";
            }
            
            findMallocManagedSize(&fun);
            
            return true;
        } /* end of runOnFunction */
       
        void identifyGlobalAddr(Function *func, LLVMContext& context, Module* module)
        {
            /* check every argument if it is a pointer type */ 
            for (auto arg_it = func->arg_begin(); arg_it != func->arg_end(); arg_it++) {
                // arg_it = Argument*, inherts Value
                Value* arg_val = dyn_cast<Value>(arg_it);
                assert(arg_val);
                {
                    Type* arg_type = arg_it->getType();
                    //if (auto ptr_type = dyn_cast<PointerType>(arg_type)) {
                    if (isa<PointerType>(arg_type) && !arg_it->hasByValAttr()) {
                        //errs() << "\t\t ===> Global Pointer " << arg_val->getName() << "\n";
                    
                        global_ptr_set.insert(arg_val->getName().str());
                        errs() << "\t\t ===> Func: " << func->getName() << ": Insert global addr : " << arg_val->getName().str()+".addr" << "\n"; 
                        global_ptr_addr_set.insert(arg_val->getName().str()+".addr");
                        global_ptr_value_set.insert(arg_val);

                        arg_val->dump();
                    }
                }
            }
            /* check every call instruction recursively */
            for (auto inst_it = inst_begin(func), it_end = inst_end(func); 
                    inst_it != it_end; inst_it++) {
                if ( auto sub_call_inst = dyn_cast<CallInst>(&(*inst_it)) ) {
                    Function* called_func = sub_call_inst->getCalledFunction();
                    if (cuda_inst_func_name != called_func->getName().str()) {
                        //identifyGlobalAddr(called_func, context, module);
                    }
                }
            }
        }

        bool isGlobalAddr(Value* value) 
        {
            bool found = false;
            while (auto inst = dyn_cast<Instruction>(value)) {
                /* get element from array */
                if (auto get_elem_ptr_inst = dyn_cast<GetElementPtrInst>(inst)) {
                    value->dump();
                    Value* ptr_op = get_elem_ptr_inst->getPointerOperand();   
                    value = ptr_op;
                }
                /* may be loading a pointer from pointer to pointer */
                else if (auto load_inst = dyn_cast<LoadInst>(inst)) {
                    errs() << __func__ <<": "; value->dump();
                    Value* ptr_op = load_inst->getPointerOperand();
                    
                    if (global_ptr_addr_set.find(ptr_op->getName().str()) != global_ptr_addr_set.end()) {
                        errs() << "\t===> load ptr op " << ptr_op->getName() << " matches with global address\n";
                        found = true;
                        break;
                    }
                    else if (isa<AllocaInst>(ptr_op)) {
                        for (auto user : ptr_op->users()) {
                            if (isa<StoreInst>(user)) {
                                StoreInst* store_inst = dyn_cast<StoreInst>(user);
                                Value* store_val = store_inst->getValueOperand();
                                errs() << __func__ << " : User Store Inst : "; store_inst->dump();
                                errs() << __func__ << " : User Store Val : "; store_val->dump();
                                
                                //auto ff = store_inst->getParent()->getParent();
                                //ff->dump();

                                if (global_ptr_addr_set.find(store_val->getName().str()) != global_ptr_addr_set.end()) {
                                    errs() << "\t===> store-load ptr op " << store_val->getName() << " matches with global address\n";
                                    found = true;
                                    return found;
                                }
                                else if (isa<GetElementPtrInst>(store_val)) {
                                    Value* addr_op = dyn_cast<GetElementPtrInst>(store_val)->getPointerOperand();
                                    if (isa<LoadInst>(addr_op)) {
                                        LoadInst *gep_load_inst = dyn_cast<LoadInst>(addr_op);
                                        Value* gep_load_addr = gep_load_inst->getPointerOperand();
                                        if (global_ptr_addr_set.find(gep_load_addr->getName().str()) != global_ptr_addr_set.end()) {
                                            errs() << "\t===> store-gep-load ptr op " << store_val->getName() << " matches with global address\n";
                                            found = true;
                                            return found;
                                        }
                                        else if (auto raddr = tid_analysis->getRealAddr(addr_op)) {
                                            value = raddr;
                                            value->dump();
                                            break;
                                        }
                                    }
                                }
                                else if (isa<LoadInst>(store_val)) {
                                    //LoadInst* store_load_inst = dyn_cast<LoadInst>(store_val);
                                    value = store_val;//store_load_inst->getPointerOperand();
                                    break;
                                }
                                else if (auto bitcast_inst = dyn_cast<BitCastInst>(store_val)) {
                                    value = bitcast_inst->getOperand(0);
                                    break;
                                }
                                else {
                                    return false;
                                }
                            }
                        }
                    }
                    else {
                        value = ptr_op;
                    }
                }
                /* may be another gep inst */
                else if (auto gep_inst = dyn_cast<GetElementPtrInst>(inst)) {
                    value = gep_inst->getPointerOperand();
                }
                else if (auto bitcast_inst = dyn_cast<BitCastInst>(inst)) {
                    value = bitcast_inst->getOperand(0);
                }
                else {
                    break;
                }
            }
            return found;
        }

        void processInstrument(Function* func, LLVMContext& context, Module* module)
        {
            if (processed_func_set.find(func->getName().str()) != processed_func_set.end()) {
                return;
            }
            else {
                processed_func_set.insert(func->getName().str());
            }
            errs() << "__func__" << ": Caller " << func->getName() << "\n";
            for (auto bb = func->begin(); bb != func->end(); bb++) {
                for (auto inst = bb->begin(); inst != bb->end(); inst++) {
                    if ( auto call_inst = dyn_cast<CallInst>(&(*inst)) ) {
                        /* Handle atomic functions */
                        errs() << "Call Instruction: "; call_inst->dump();
                        if (call_inst->getCalledOperand()->getName().str().find("_ZL9atomic") != string::npos) {
                            errs() << "Atomic functions detected\n";
                            Value* addr_param = call_inst->getArgOperand(0);
                            atomicInsertInstCode(call_inst, context, module);
                            continue;
                        }
                        
                        for (auto call_op = call_inst->op_begin(); /* op is Use* Type */
                                call_op != call_inst->op_end(); 
                                call_op++) 
                        {
                            Value* val = call_op->get();
                            Function* callee_func = module->getFunction(val->getName());
                            if (callee_func) {
                                //errs() << __func__ << ": Callee ";
                                if (callee_func->getName().str().find("llvm.nvvm.read.ptx.sreg") != string::npos
                                    || callee_func->getName().str().find("__cuda_builtin") != string::npos) {
                                    errs() << ": (name " << callee_func->getName() << ")\n";
                                    if (callee_func->getName().str().find("builtin_blockIdx") != string::npos && 
                                        callee_func->getName().str().find("builtin_yEv") != string::npos) {
                                        blockIdx_y_str = callee_func->getName().str();
                                        errs() << "\t found blockIdx.y: " << blockIdx_y_str << "\n";
                                    }    
                                    else if (callee_func->getName().str().find("builtin_blockDim") != string::npos && 
                                        callee_func->getName().str().find("builtin_yEv") != string::npos) {
                                        blockDim_y_str = callee_func->getName().str();
                                        errs() << "\t found blockDim.y: " << blockDim_y_str << "\n";
                                    }
                                    if (callee_func->getName().str().find("builtin_threadIdx") != string::npos && 
                                        callee_func->getName().str().find("builtin_yEv") != string::npos) {
                                        threadIdx_y_str = callee_func->getName().str();
                                        errs() << "\t found threadIdx.y: " << threadIdx_y_str << "\n";
                                    }

                                    break;
                                }
                                //call_inst->dump();
                                processInstrument(callee_func, context, module);
                            }
                        }
                    }
                    else if ( auto load_inst = dyn_cast<LoadInst>(&(*inst)) ) {
                        Value* ptr_value = load_inst->getPointerOperand();            
                        AAResults &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
                        if (isGlobalAddr(ptr_value)) {
                            errs() << "Insert instrumenting function after ";
                            load_inst->dump(); 
                            if (processed_mem_inst_set.find(load_inst) != processed_mem_inst_set.end()) {
                                continue;
                            }
                            loadInsertInstCode(load_inst, context, module, "_Z10deviceRecvPvi");
                        }
                        errs() << "\n\n";
                    }
                    else if ( auto store_inst = dyn_cast<StoreInst>(&(*inst)) ) {
                        Value* ptr_value = store_inst->getPointerOperand();            
                        AAResults &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
                        //errs() << "ptr op: " << ptr_value->getName() << "\n";
                        //if (global_ptr_set.find(ptr_value->getName().str()) != global_ptr_set.end()) {
                        if (isGlobalAddr(ptr_value)) {
                            errs() << "Insert instrumenting function after ";
                            store_inst->dump(); 
                            if (processed_mem_inst_set.find(store_inst) != processed_mem_inst_set.end()) {
                                continue;
                            }
                            storeInsertInstCode(store_inst, context, module, "_Z10deviceRecvPvii");
                        }
                        errs() << "\n\n";
                    }
                }
            }
        }
        
        /* Special handling for atomic functions */
        void atomicInsertInstCode(CallInst* calling_inst, LLVMContext& context, Module* module)
        {
            Function* inst_func;
            IRBuilder<> builder(calling_inst->getNextNonDebugInstruction());
            Value* addr_val = calling_inst->getArgOperand(0);
            Value* ir_ptr = builder.CreateBitCast(addr_val, Type::getInt8PtrTy(context));

            inst_func = module->getFunction(devRecv_Func_Name);
            /* get the type we are accessing */
            Type* base = addr_val->getType();
            DataLayout* data_layout = new DataLayout(module);
            int loadSize = (int)data_layout->getTypeStoreSize(base);
            ConstantInt* ir_loadSize = builder.getInt32(loadSize);
            ConstantInt* ir_acc_num = builder.getInt32(-1);
            
            /* create args array */
            Value* args[] = {ir_ptr, ir_loadSize, ir_acc_num};

            /* create a call and insert it before the store instruction */
            CallInst* call_inst = builder.CreateCall(inst_func, args);
            assert(call_inst);
        }

        void loadInsertInstCode(LoadInst *load_inst, LLVMContext& context, Module* module, StringRef func_name, int acc_num = -1)
        {
            #if INSERT_BEFORE
            IRBuilder<> builder(load_inst);
            #else
            IRBuilder<> builder(load_inst->getNextNonDebugInstruction());
            #endif
            THREAD_ACCESS_TYPE_E access_type = ACC_NOT_FOUND;
            Value* final_scale = NULL;

            /* get operands */
            Value* ptr = load_inst->getPointerOperand();
            Value* ir_ptr = builder.CreateBitCast(ptr, Type::getInt8PtrTy(context));
            
            /* analyze scale and offset */
            vector<pair<llvm::Instruction::BinaryOps,Value*>> scale;
            vector<pair<llvm::Instruction::BinaryOps,Value*>> offset;
            access_type = tid_analysis->analyze_gep_for_global(dyn_cast<GetElementPtrInst>(ptr), scale, offset);
            if (access_type == ACC_FULL_THREAD_ID) {
                final_scale = tid_analysis->get_final_scale_1d(context, builder, scale);
            }
            else {
                vector<pair<llvm::Instruction::BinaryOps,Value*>> scale_block_x;
                vector<pair<llvm::Instruction::BinaryOps,Value*>> scale_block_y;
                access_type = tid_analysis->analyze_gep_for_global_2d(dyn_cast<GetElementPtrInst>(load_inst->getPointerOperand()), 
                                                                      scale_block_x, 
                                                                      scale_block_y);
                if (access_type == ACC_BLOCK_ID_X) {
                    final_scale = tid_analysis->get_final_scale_1d(context, builder, scale_block_x);
                }
                else if (access_type == ACC_BLOCK_ID_Y) {
                    final_scale = tid_analysis->get_final_scale_1d(context, builder, scale_block_y);
                }
            }
            errs() << "----------> access type " << THREAD_ACCESS_TYPE_STR[access_type] << "\n";
            if (final_scale) {
                errs() << "----------> final scale "; final_scale->dump();
            }

            /* get instrumenting code */
            FunctionType* inst_func_type = getInstFuncType(context);
            //FunctionCallee func_callee = module->getOrInsertFunction(func_name, inst_func_type);
            
            Function* inst_func;
            ConstantInt* ir_scale_factor;
            if (access_type == ACC_FULL_THREAD_ID) {
                inst_func = module->getFunction(update_PST_1D_Func_Name);
            }
            else if (access_type == ACC_BLOCK_ID_X) {
                inst_func = module->getFunction(update_PST_2D_Block_X_Func_Name);
                errs() << "Instrumentation Function : " << update_PST_2D_Block_X_Func_Name << "\n";
            }
            else if (access_type == ACC_BLOCK_ID_Y) {
                inst_func = module->getFunction(update_PST_2D_Block_Y_Func_Name);
                errs() << "Instrumentation Function : " << update_PST_2D_Block_Y_Func_Name << "\n";
            }
            else {
                inst_func = module->getFunction(devRecv_Func_Name);
            }
            //FIXME
            inst_func = module->getFunction(devRecv_Func_Name);

            /* get the type we are accessing */
            Type* base = load_inst->getType();
            DataLayout* data_layout = new DataLayout(module);
            int loadSize = (int)data_layout->getTypeStoreSize(base);
            ConstantInt* ir_loadSize = builder.getInt32(loadSize);
            ConstantInt* ir_acc_num = builder.getInt32(acc_num);
            
            /* Cast to Integer 32 */
            if (access_type != ACC_NOT_FOUND) {
                final_scale = builder.CreateIntCast(final_scale, Type::getInt32Ty(context), true /*signed*/);
            }

            /* create args array */
            Value* args[] = {ir_ptr, ir_loadSize, (access_type == ACC_NOT_FOUND) ? ir_acc_num : final_scale};
            if (access_type == ACC_NOT_FOUND) {
                //errs() << "acc num: "; ir_acc_num->dump();
                //inst_func->dump();
            }

            /* create a call and insert it before the store instruction */
            //CallInst* call_inst = builder.CreateCall(func_callee, args);
            //assert(cuda_inst_func);
            
            CallInst* call_inst = builder.CreateCall(inst_func, args);
            assert(call_inst);
            
            //auto thisbb = load_inst->getParent();
            //thisbb->dump();

            //BitCastInst* ptr_cast_inst = dyn_cast<BitCastInst>(ir_ptr);
            //ptr_cast_inst->insertAfter(load_inst);
            //call_inst->insertAfter(ptr_cast_inst);
            
            //call_inst->dump();
            num_of_inserted_load++;
        }
 
        void storeInsertInstCode(StoreInst *store_inst, LLVMContext& context, Module* module, StringRef func_name, int acc_num = -1)
        {
            #if INSERT_BEFORE
            IRBuilder<> builder(store_inst);
            #else
            IRBuilder<> builder(store_inst->getNextNonDebugInstruction());
            #endif
            //IRBuilder<> builder(context);
            THREAD_ACCESS_TYPE_E access_type = ACC_NOT_FOUND;
            Value* final_scale = NULL;
            Value* refered_addr = NULL;

            /* get operands */
            Value* val = store_inst->getValueOperand();
            Value* ptr = store_inst->getPointerOperand();
            Value* ir_ptr = builder.CreateBitCast(ptr, Type::getInt8PtrTy(context));
            
            /* analyze scale and offset */
            vector<pair<llvm::Instruction::BinaryOps,Value*>> scale;
            vector<pair<llvm::Instruction::BinaryOps,Value*>> offset;
            if (isa<GetElementPtrInst>(ptr)) {
                access_type = tid_analysis->analyze_gep_for_global(dyn_cast<GetElementPtrInst>(ptr), scale, offset);
            }
            else if (isa<LoadInst>(ptr)){
                refered_addr = tid_analysis->getRealAddr(ptr);
                ptr = refered_addr;
                //assert(isa<GetElementPtrInst>(ptr));
                if (isa<GetElementPtrInst>(ptr)) {
                    access_type = tid_analysis->analyze_gep_for_global(dyn_cast<GetElementPtrInst>(ptr), scale, offset);
                }
            }
            
            /* if it is not full thread access, try 2d case */
            if (access_type == ACC_FULL_THREAD_ID) {
                final_scale = tid_analysis->get_final_scale_1d(context, builder, scale);
            }
            else {
                vector<pair<llvm::Instruction::BinaryOps,Value*>> scale_block_x;
                vector<pair<llvm::Instruction::BinaryOps,Value*>> scale_block_y;
                access_type = tid_analysis->analyze_gep_for_global_2d(dyn_cast<GetElementPtrInst>(ptr), 
                                                                      scale_block_x, 
                                                                      scale_block_y);
                if (access_type == ACC_BLOCK_ID_X) {
                    final_scale = tid_analysis->get_final_scale_1d(context, builder, scale_block_x);
                }
                else if (access_type == ACC_BLOCK_ID_Y) {
                    final_scale = tid_analysis->get_final_scale_1d(context, builder, scale_block_y);
                }
            }
            errs() << "----------> access type " << THREAD_ACCESS_TYPE_STR[access_type] << "\n";
            if (access_type != ACC_NOT_FOUND && final_scale) {
                errs() << "----------> final scale "; final_scale->dump();
            }
            //DataLayout test_dl(module);
            //tid_analysis->test(ptr, test_dl);
            //tid_analysis->mem_loc_test(store_inst);
            
            Function* inst_func;
            ConstantInt* ir_scale_factor;
            if (access_type == ACC_FULL_THREAD_ID) {
                inst_func = module->getFunction(update_PST_1D_Func_Name);
            }
            else if (access_type == ACC_BLOCK_ID_X) {
                inst_func = module->getFunction(update_PST_2D_Block_X_Func_Name);
                errs() << "Instrumentation Function : " << update_PST_2D_Block_X_Func_Name << "\n";
            }
            else if (access_type == ACC_BLOCK_ID_Y) {
                inst_func = module->getFunction(update_PST_2D_Block_Y_Func_Name);
                errs() << "Instrumentation Function : " << update_PST_2D_Block_Y_Func_Name << "\n";
            }
            else {
                inst_func = module->getFunction(devRecv_Func_Name);
            } 
            //FIXME
            inst_func = module->getFunction(devRecv_Func_Name);

            /* get instrumenting code */
            FunctionType* inst_func_type = getInstFuncType(context);
            //FunctionCallee func_callee = module->getOrInsertFunction(func_name, inst_func);
            assert(inst_func);

            /* get the type we are accessing */
            Type* base = store_inst->getValueOperand()->getType();
            DataLayout* data_layout = new DataLayout(module);
            int storeSize = (int)data_layout->getTypeStoreSize(base);
            ConstantInt* ir_storeSize = builder.getInt32(storeSize);
            ConstantInt* ir_acc_num = builder.getInt32(acc_num);
            
            /* Cast to Integer 32 */
            if (access_type != ACC_NOT_FOUND) {
                final_scale = builder.CreateIntCast(final_scale, Type::getInt32Ty(context), true /*signed*/);
            }

            /* create args array */
            Value* args[] = {ir_ptr, ir_storeSize, (final_scale == NULL) ? ir_acc_num : final_scale};

            /* create a call and insert it before the store instruction */
            CallInst *call_inst = builder.CreateCall(inst_func, args);
            assert(call_inst);
            //dyn_cast<BitCastInst>(ir_ptr)->insertAfter(store_inst);
            //call_inst->insertAfter(dyn_cast<BitCastInst>(ir_ptr));
            
            //call_inst->dump();
            num_of_inserted_store++;
        }

        FunctionType* getInstFuncType(LLVMContext& context) {
            Type* retType = Type::getVoidTy(context);
            vector<Type*> args { 
                Type::getInt8PtrTy(context),
                Type::getInt32Ty(context)
            };
            FunctionType* instrument = FunctionType::get(retType, args, false);
            return instrument;
        }
        
        FunctionType* getStatTableAllocFuncType(LLVMContext& context) {
            Type* retType = Type::getVoidTy(context);
            vector<Type*> args { 
                Type::getInt64PtrTy(context),
                Type::getInt64Ty(context)
            };
            FunctionType* instrument = FunctionType::get(retType, args, false);
            return instrument;
        }
 
        /* find cudaMalloc size */
        void findMallocManagedSize(Function* func)
        {
            LLVMContext& context = func->getContext();
            for (auto inst = inst_begin(func); inst != inst_end(func); inst++) {
                if (!isa<CallInst>(&*inst) && !isa<InvokeInst>(&*inst)) 
                    continue;
                
                if (auto call_inst = dyn_cast<CallInst>(&*inst)) {
                    auto called_func = call_inst->getCalledValue();
                    if (called_func->getName().str().find(cudaMallocManagedName) != string::npos
                        && func->getName().str().find(cudaMallocManagedName) == string::npos) {
                    //errs() << __func__ << ": " << called_func->getName() << "\n";
                    //if (called_func->getName().str().find("cudaMallocManaged") != string::npos) {
                        errs() << "--- Function: " << called_func->getName() << "\n";
                        call_inst->dump();
                        
                        Value* ptr_addr_val = call_inst->getArgOperand(0);
                        Value* alloc_size_val = call_inst->getArgOperand(1);

                        if (auto cnst = dyn_cast<Constant>(alloc_size_val)) {
                            cnst->dump();
                        }
                        BitCastInst* cast_ptr_inst = new BitCastInst(ptr_addr_val, Type::getInt64PtrTy(context));
                        cast_ptr_inst->insertAfter(call_inst);
                        Value* cast_ptr_val = dyn_cast<Value>(cast_ptr_inst);
                        assert(cast_ptr_val);
                        cast_ptr_inst->dump();

                        /* get the value of ptr and size, push them to instrumenting function */
                        FunctionType* instrument_func_type = getStatTableAllocFuncType(func->getContext());
                        Function* inst_func = func->getParent()->getFunction(statTableFuncName);
                        assert(inst_func);
                        //Value* args[] = {ptr_addr_val, alloc_size_val};
                        Value* args[] = {cast_ptr_val, alloc_size_val};

                        CallInst* instrument_call_inst = CallInst::Create(instrument_func_type, inst_func, args);

                        //instrument_call_inst->insertAfter(call_inst);
                        instrument_call_inst->insertAfter(cast_ptr_inst);
                    } 
                }
                else if (auto invoke_inst = dyn_cast<InvokeInst>(&*inst)) {
                    auto called_func = invoke_inst->getCalledOperand();
                    //errs() << "Invoke --- " << called_func->getName() << "\n";
                    //called_func->dump();
                    if (called_func->getName().str().find(cudaMallocManagedName) != string::npos) {
                        Value* ptr_addr_val = invoke_inst->getArgOperand(0);
                        Value* alloc_size_val = invoke_inst->getArgOperand(1);
                        
                        //errs() << "third arg : "; invoke_inst->getNormalDest()->dump();

                        if (auto cnst = dyn_cast<Constant>(alloc_size_val)) {
                            cnst->dump();
                        }
                        BitCastInst* cast_ptr_inst = new BitCastInst(ptr_addr_val, Type::getInt64PtrTy(context));
                        //cast_ptr_inst->insertAfter(invoke_inst);
                        cast_ptr_inst->insertBefore(invoke_inst->getNormalDest()->getFirstNonPHI());
                        Value* cast_ptr_val = dyn_cast<Value>(cast_ptr_inst);
                        assert(cast_ptr_val);
                        cast_ptr_inst->dump();

                        /* get the value of ptr and size, push them to instrumenting function */
                        FunctionType* instrument_func_type = getStatTableAllocFuncType(func->getContext());
                        Function* inst_func = func->getParent()->getFunction(statTableFuncName);
                        assert(inst_func);
                        //Value* args[] = {ptr_addr_val, alloc_size_val};
                        Value* args[] = {cast_ptr_val, alloc_size_val};

                        CallInst* instrument_call_inst = CallInst::Create(instrument_func_type, inst_func, args);

                        //instrument_call_inst->insertAfter(call_inst);
                        instrument_call_inst->insertAfter(cast_ptr_inst);
                    }
                }
            }
        } /* end of findMallocManagedSize */
       
        int get_line_acc_num_for_mem_inst(GetElementPtrInst* gep_inst, DataLayout* data_layout)
        {
            Value* src_addr = NULL;
            Value* src_index = NULL;
            Value* src_offset = NULL;
            Value* src_scale = NULL;
            bool is_src_index_cnst;
            bool is_src_offset_cnst;
            bool is_src_scale_cnst;
            int access_size_bytes;
            int offset_int_val;
            int scale_int_val;

            find_src_addr_and_index_by_getelementptr(
                gep_inst,
                src_addr,
                src_index,
                is_src_index_cnst,
                src_offset,
                is_src_offset_cnst,
                src_scale,
                is_src_scale_cnst);
            
            if (is_src_index_cnst) {
                return 32;
            }

            if (!is_src_index_cnst && src_index && !is_loading_thread_id(src_index)) {
                return 1;
            }

            if (src_scale == NULL) {
                scale_int_val = 1;
            }
            else {
                scale_int_val = dyn_cast<ConstantInt>(src_scale)->getSExtValue();
            }
            if (src_offset == NULL) {
                offset_int_val = 0;
            }
            else {
                offset_int_val = dyn_cast<ConstantInt>(src_offset)->getSExtValue();
            }
            
            Type* type = gep_inst->getSourceElementType();
            access_size_bytes = data_layout->getTypeAllocSize(type);

            errs() << gep_inst->getName() << ": offset " << offset_int_val << ", scale " << scale_int_val << "\n";

            return (((QUANT_LINE_SIZE/access_size_bytes)-offset_int_val)/scale_int_val);
        }
        
        void process_dup_for_load_store_basic_block (BasicBlock* bb, Value* init_inst, Value* mem_ptr, int mem_size, DataLayout* data_layout, AAResults &aa_results)
        {
            BasicBlock::iterator bb_inst = bb->begin();
            if (init_inst) {
                while (&(*bb_inst) != init_inst) bb_inst++;
                bb_inst++;
            }
            for (; bb_inst != bb->end(); bb_inst++) {
                if (isa<LoadInst>(bb_inst)) {
                    int load_size = data_layout->getTypeStoreSize(cast<LoadInst>(bb_inst)->getType());
                    if (auto alias_res 
                            = aa_results.alias(mem_ptr, mem_size, cast<LoadInst>(bb_inst)->getPointerOperand(), load_size) != AliasResult::NoAlias) {
                        /* may alias, skip instrumenting */
                        processed_mem_inst_set.insert(&*bb_inst);
                    }
                }
                else if (isa<StoreInst>(bb_inst)) {
                    int store_size = data_layout->getTypeStoreSize(cast<StoreInst>(bb_inst)->getValueOperand()->getType());
                    if (auto alias_res 
                            = aa_results.alias(mem_ptr, mem_size, cast<StoreInst>(bb_inst)->getPointerOperand(), store_size) != AliasResult::NoAlias) {
                        /* may alias, skip instrumenting */
                        processed_mem_inst_set.insert(&*bb_inst);
                    }
                }
            }
        }

        /* Assuming all IR has been processed by Constant Propagation and Common Subexpression Elimination */
        void process_temporal_space_dup_for_load_store( BasicBlock* bb, 
                                                        Instruction* mem_inst,  
                                                        DataLayout* data_layout ) 
        {   
            int mem_size;
            Value* mem_ptr = nullptr;
            AAResults &aa_results = getAnalysis<AAResultsWrapperPass>().getAAResults();

            BasicBlock::iterator bb_inst = bb->begin();
            while (&(*bb_inst) != cast<Value>(mem_inst)) bb_inst++;
            bb_inst++; 
            
            if (isa<LoadInst>(mem_inst)) {
                mem_ptr = cast<LoadInst>(mem_inst)->getPointerOperand();
                mem_size = data_layout->getTypeStoreSize(cast<LoadInst>(mem_inst)->getType());
            }
            else if (isa<StoreInst>(mem_inst)) {
                mem_ptr = cast<StoreInst>(mem_inst)->getPointerOperand();
                mem_size = data_layout->getTypeStoreSize(cast<StoreInst>(mem_inst)->getValueOperand()->getType());
            }
            
            // now bb_inst should be the next one instruction after mem_inst
            /* now process memory alias */
            process_dup_for_load_store_basic_block (bb, cast<Value>(mem_inst), mem_ptr, mem_size, data_layout, aa_results);
            

            /* now process the following basic blocks */
            SmallVector<BasicBlock*, 16> work_list_vec;
            SmallVector<BasicBlock*, 16> next_work_list_vec;
            for (auto succ_bb_it = succ_begin(bb); succ_bb_it != succ_end(bb); succ_bb_it++) {
                work_list_vec.push_back(*succ_bb_it);
            }
            while (!work_list_vec.empty()) {
                
            }
            
        } /* end of process_temporal_dup_for_load_store */

        void find_src_addr_and_index_by_getelementptr(GetElementPtrInst* gep_inst,
                                                        Value*& src_addr,
                                                        Value*& src_index,
                                                        bool& is_src_index_cnst,
                                                        Value*& src_offset,
                                                        bool& is_src_offset_cnst, 
                                                        Value*& src_scale,
                                                        bool& is_src_scale_cnst)
        {
            Value* ptr_addr = gep_inst->getPointerOperand();
            Value* idx = gep_inst->idx_begin()->get();
            
            if (gep_inst->getNumIndices() > 1) {
                return;
            }
            
            errs() << "--- handle gep instruction ---\n";
            gep_inst->dump();
            idx->dump();

            /* get src addr */ 
            auto load_src_addr_inst = dyn_cast<LoadInst>(ptr_addr);
            src_addr = load_src_addr_inst->getPointerOperand();

            /* get src index */
            auto get_idx_inst = dyn_cast<Instruction>(idx);
            if (get_idx_inst == NULL) {
                ConstantInt* const_int = dyn_cast<ConstantInt>(idx);
                assert(const_int);
                is_src_index_cnst = true;
                src_index = idx;
                errs() << "\tindex is constant: " << const_int->getSExtValue() << "\n";
            }
            else {
                while (get_idx_inst) {
                    /* let us suppose idx = a*tid + b 
                        and only consider constants now ... */
                    if (auto sext_inst = dyn_cast<SExtInst>(get_idx_inst)) {
                        get_idx_inst = dyn_cast<Instruction>(sext_inst->getOperand(0));
                    }
                    else if (auto zext_inst = dyn_cast<ZExtInst>(get_idx_inst)) {
                        get_idx_inst = dyn_cast<Instruction>(zext_inst->getOperand(0));
                    }
                    else if (auto call_inst = dyn_cast<CallInst>(get_idx_inst)) {
                        is_src_index_cnst = false;
                        src_index = dyn_cast<Value>(get_idx_inst);
                        get_idx_inst = NULL;
                    }
                    /* arithmetic operations to calculate index, just tracing back... */
                    else if (auto bin_inst = dyn_cast<BinaryOperator>(get_idx_inst)) {
                        // might be add or mul
                        
                        Value* op1 = bin_inst->getOperand(0);
                        Value* op2 = bin_inst->getOperand(1);
                        /* if both are not constant, ignore */
                        ConstantInt* cnst_op1 = dyn_cast<ConstantInt>(op1);
                        ConstantInt* cnst_op2 = dyn_cast<ConstantInt>(op2);
                        
                        switch (bin_inst->getOpcode()) 
                        {
                            case llvm::Instruction::Add:
                            {
                                if (!cnst_op1 && !cnst_op2) {
                                    is_src_offset_cnst = false;
                                    is_src_scale_cnst = false;
                                    errs() << "\tAdd inst: " << op1->getName() << " : " << op2->getName() << "\n";
                                }
                                else if (cnst_op1 && !cnst_op2) {
                                    get_idx_inst = dyn_cast<Instruction>(op2); 
                                    src_offset = op1;
                                    is_src_offset_cnst = true;
                                    errs() << "\tAdd inst: " << cnst_op1->getValue() << " : " << op2->getName() << "\n";
                                }
                                else if (!cnst_op1 && cnst_op2) {
                                    get_idx_inst = dyn_cast<Instruction>(op1); 
                                    src_offset = op2;
                                    is_src_offset_cnst = true;
                                    errs() << "\tAdd inst: " << op1->getName() << " : " << cnst_op2->getValue() << "\n";
                                }
                                break;
                            }
                            case llvm::Instruction::Mul:
                            {
                                if (cnst_op1 && !cnst_op2) {
                                    get_idx_inst = dyn_cast<Instruction>(op2); 
                                    src_scale = op1;
                                    is_src_scale_cnst = true;
                                    errs() << "\tMul inst: " << cnst_op1->getValue() << " : " << op2->getName() << "\n";
                                }
                                else if (!cnst_op1 && cnst_op2) {
                                    get_idx_inst = dyn_cast<Instruction>(op1); 
                                    src_scale = op2;
                                    is_src_scale_cnst = true;
                                    errs() << "\tMul inst: " << op1->getName() << " : " << cnst_op2->getValue() << "\n";
                                }
                                else if (!cnst_op1 && !cnst_op2) {
                                    is_src_index_cnst = false;
                                    is_src_scale_cnst = false;
                                    src_index = op1;
                                    src_scale = op2;
                                    get_idx_inst = NULL;
                                    errs() << "\tMul inst: " << op1->getName() << " : " << op2->getName() << "\n";
                                }
                                break;
                            }
                            default:
                                break;
                        } /* end of swith case */
                    } /* end of else if of casting get_idx_inst to bin_inst */
                    else if (auto load_inst = dyn_cast<LoadInst>(get_idx_inst)) {
                        src_index = load_inst->getPointerOperand();
                        is_src_index_cnst = false;
                        get_idx_inst = NULL;
                        errs() << "\tGep base index: " << src_index->getName() << "\n";                        
                    } /* end of else if of casting get_idx_inst to load_inst */
               } /* end of while loop of getting src index */
           }  

        } /* end of find_src_addr_and_index_by_getelementptr */
        
        bool is_loading_thread_id(Value* base) 
        {   
            errs() << __func__ << ":";
            base->dump();
            if (auto call_inst = dyn_cast<CallInst>(base)) {
                return is_composed_by_thread_id(base);
            }
            else if (auto alloc_base_inst = dyn_cast<AllocaInst>(base)) {
                //Value* load_loc = base_load_inst->getPointerOperand();
                /* find the store instruction that stores thread id to load_loc */
                for (auto user = base->user_begin(); user != base->user_end(); user++) {
                    errs() << "\t"; user->dump();
                    if (auto used_store_inst = dyn_cast<StoreInst>(*user)) { /* *user returns the pointer of user */
                        if (is_composed_by_thread_id(used_store_inst->getValueOperand())) {
                            return true;
                        }
                    }
                }
            }
            return false;
        }
        
        bool is_1d_thread_id (Value* val)
        {   
            /* e.g. store i32 %add, i32* %tid */
            bool ret = false;
            if (auto alloc_val_inst = dyn_cast<AllocaInst>(val)) {
                for (auto user = val->user_begin(); user != val->user_end(); user++) {
                    if (auto used_store_inst = dyn_cast<StoreInst>(*user)) { /* *user returns the pointer of user */
                        auto store_val = used_store_inst->getValueOperand();
                        val = store_val;            
                    }
                } 
            }
            if (auto zext_inst = dyn_cast<ZExtInst>(val)) {
                val = zext_inst->getOperand(0);
            }
            if (auto sext_inst = dyn_cast<SExtInst>(val)) {
                val = sext_inst->getOperand(0);
            }
 
            if (auto bin_inst = dyn_cast<BinaryOperator>(val)) {
                Value* op1 = bin_inst->getOperand(0);
                Value* op2 = bin_inst->getOperand(1);
                BinaryOperator* bin_op = NULL;
                CallInst* get_tid_call_inst = NULL;
                CallInst* get_bid_call_inst = NULL;
                CallInst* get_bdim_call_inst = NULL;

                if (bin_op = dyn_cast<BinaryOperator>(op1)) {
                    get_tid_call_inst = dyn_cast<CallInst>(op2); 
                }
                else if (bin_op = dyn_cast<BinaryOperator>(op2)) {
                    get_tid_call_inst = dyn_cast<CallInst>(op1); 
                }
                get_bid_call_inst = dyn_cast<CallInst>(bin_op->getOperand(0));
                get_bdim_call_inst = dyn_cast<CallInst>(bin_op->getOperand(1));

                ret |= ((get_tid_call_inst->getName().str().find(threadIdx_x_str) != string::npos) 
                        && (get_bid_call_inst->getName().str().find(blockIdx_x_str) != string::npos) 
                        && (get_bdim_call_inst->getName().str().find(blockDim_x_str) != string::npos));

                return ret;
            }

            return ret;
        }

        bool is_composed_by_thread_id(Value* val)
        {
            /* the value may be calculated by several binary operations 
                but eventually there sho*/
            if (auto bin_inst = dyn_cast<BinaryOperator>(val)) {
                Value* op1 = bin_inst->getOperand(0);
                Value* op2 = bin_inst->getOperand(1);
                return is_composed_by_thread_id(op1) || is_composed_by_thread_id(op2); 
            }
            else if (auto call_inst = dyn_cast<CallInst>(val)) {
                auto called_func = call_inst->getCalledValue();
                errs() << __func__ << ": " << called_func->getName() << "\n";
                if (called_func->getName().str().find(threadIdx_x_str) != string::npos) {
                    return true;
                }
            }
            return false;
        }

        //BasicBlock* get_basic_block_by_instruction(Function* func, Instruction*)
    private:
        unordered_set<Value*> processed_mem_inst_set;    
        
        /* helper functions */
        bool name_contains (Value* val, string& _str) {
            return val->getName().str().find(_str) != string::npos;
        }
    
    friend class ThreadIdxAnalysis;

    }; /* end of MemAccInst */

} /* end of namespace */

/* ID initial value is not important at all, LLVM uses the address of ID to identify */
char MemAccInst::ID = 0;
//string MemAccInst::cudaMallocManagedName = "_ZL17cudaMallocManagedIiE9cudaErrorPPT_mj";
string MemAccInst::cudaMallocManagedName = "cudaMallocManaged";
string MemAccInst::statTableFuncName = "_Z24mem_acc_stat_table_allocPmm";
string MemAccInst::devRecvWithLineFuncName = "_Z24deviceRecvWithLineAccNumPvii";
string MemAccInst::update_PST_1D_Func_Name = "_Z13update_PST_1dPvii";
string MemAccInst::threadIdx_x_str = "__cuda_builtin_threadIdx_t17__fetch_builtin_xEv";
string MemAccInst::blockIdx_x_str = "__cuda_builtin_blockIdx_t17__fetch_builtin_xEv";
string MemAccInst::blockDim_x_str = "__cuda_builtin_blockDim_t17__fetch_builtin_xEv";


/* Register the class (command line argument, name, only looks at CFG, analysis pass) */
static RegisterPass<MemAccInst> X("MemAccInst", "Memory Access Instrument Pass", false, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [](const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
        PM.add(new MemAccInst());
    }
);
