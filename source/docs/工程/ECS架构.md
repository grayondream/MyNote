# ECS架构
## 1 简介
&emsp;&emsp;在当今快速发展的软件开发领域，游戏开发、实时模拟等场景对系统的性能、灵活性和可扩展性提出了极高的要求。传统的面向对象架构在面对复杂且动态变化的实体时，往往会出现代码耦合度高、扩展性差等问题。​

&emsp;&emsp;ECS（Entity - Component - System，实体 - 组件 - 系统）架构作为一种新兴的DOD架构模式，凭借其独特的设计理念，能够有效解决这些难题，在众多领域得到了广泛的应用。本文将对 ECS 架构进行全面、深入的技术剖析。

![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/0_6XTw569bOFZfmS-8.png)

&emsp;&emsp;ECS 架构核心概念：
- 实体（Entity）​。实体是 ECS 架构中最基础的概念，它代表着游戏或应用中的一个独立个体，例如游戏中的一个角色、一件物品、一个敌人等。从本质上来说，实体本身并没有任何实际的行为和属性，它更像是一个标识符或者说是一个容器，用于关联各个组件。​
在实现上，实体通常可以用一个唯一的 ID 来表示，这个 ID 就如同实体的 “身份证”，系统通过这个 ID 来识别和管理不同的实体。
- 组件（Component）​。组件是用于存储数据的结构，它只包含数据，不包含任何行为逻辑。每个组件都对应着实体的某一种特定属性或状态，比如位置组件可以存储实体的坐标信息，速度组件可以存储实体的移动速度，生命值组件可以存储实体的健康状况等。​一个实体可以拥有多个不同的组件，这些组件共同构成了实体的完整特征。例如，一个游戏角色实体可以同时拥有位置组件、速度组件、生命值组件和武器组件等。
- 系统（System）。系统是 ECS 架构中负责处理行为逻辑的部分，它通过操作实体所拥有的组件数据来实现各种功能。系统不会直接作用于实体本身，而是根据实体所拥有的组件类型来筛选出需要处理的实体，并对这些实体的组件数据进行操作。​
比如，移动系统会筛选出拥有位置组件和速度组件的实体，然后根据速度组件中的数据来更新位置组件中的坐标信息；碰撞检测系统会筛选出拥有位置组件和碰撞体积组件的实体，通过计算它们的位置和碰撞体积来判断是否发生碰撞。

ECS 架构的工作流程可以简单概括为以下几个步骤：​
- 创建实体：根据业务需求创建相应的实体，并为每个实体分配唯一的 ID。​
- 添加组件：为实体添加所需的组件，组件中包含了实体的具体数据。一个实体可以添加多个不同的组件，以描述其各种属性和状态。​
- 系统处理：各个系统按照自身的逻辑，对拥有特定组件组合的实体进行处理。系统会遍历所有符合条件的实体，读取或修改它们组件中的数据，从而实现各种功能，如移动、碰撞检测、渲染等。​
- 动态更新：在应用运行过程中，可以动态地为实体添加或移除组件，实体的属性和行为也会随之发生变化。同时，系统也可以根据需要进行动态的加载或卸载，以适应不同的场景需求。

## 2 ECS vs OOP
&emsp;&emsp;在传统的面向对象架构中，一个对象通常同时包含数据和行为。例如，一个 “角色” 类会包含角色的位置、生命值等数据，以及移动、攻击等方法。这种架构的优点是符合人们的直觉思维，便于理解和设计简单的系统。​然而，当系统变得复杂时，传统面向对象架构会暴露出一些问题。一方面，类之间的继承关系会导致代码耦合度高，一旦父类发生变化，子类可能会受到很大影响，不利于系统的维护和扩展。另一方面，当需要对对象的行为进行修改时，往往需要修改类的内部实现，违反了封装原则。而相比之下ECS 架构优势​：
- **低耦合高内聚**：ECS 架构将数据和行为分离，组件只负责存储数据，系统只负责处理行为，实体只是组件的集合。这种分离使得各部分之间的耦合度大大降低，每个部分可以独立地进行开发、测试和维护，提高了代码的内聚性。​
- **更好的可扩展性**：在 ECS 架构中，当需要添加新的功能时，只需要添加相应的组件和系统即可，不需要修改现有的实体和其他系统。例如，要给游戏角色添加 “魔法” 属性，只需创建一个 “魔法值” 组件，并为需要的实体添加该组件，再创建一个处理魔法相关行为的系统即可，不会对现有的移动、攻击等系统产生影响。​
- **高效的性能**：由于系统只处理拥有特定组件的实体，避免了对无关实体的处理，提高了处理效率。此外，组件的数据存储方式通常是连续的，有利于 CPU 缓存，减少了缓存未命中的情况，从而提升了系统的性能，尤其在处理大量实体的场景下，优势更加明显。​
- **更好的并行性**：ECS 架构中，不同的系统处理不同的组件组合，且系统之间没有直接的依赖关系，这使得多个系统可以并行执行，充分利用多核 CPU 的性能，提高系统的运行效率。

&emsp;&emsp;实体组件架构（ECS）作为游戏开发等领域常用的架构模式，虽在灵活性和性能方面有一定优势，但也存在不少缺点：
- **学习和理解门槛较高**。相比传统的面向对象架构，ECS 的思维模式有很大差异，它将实体、组件、系统进行了明确分离，这种分离式的设计理念对于习惯了类继承和对象封装的开发者来说，需要花费较多时间去适应和掌握。开发者不仅要理解三者各自的职责，还要搞清楚它们之间的交互机制，比如系统如何通过组件来操作实体，实体如何动态添加或移除组件等，这无疑增加了团队的学习成本，尤其是对于新手开发者，可能需要更长时间才能熟练运用；
- **调试和排查问题难度大**。在 ECS 中，实体的状态由多个组件共同决定，而系统则是对组件进行操作的逻辑单元。当某个实体出现异常行为时，问题可能并非集中在某一个地方，而是分散在多个相关的组件和系统中。例如，一个角色移动异常，可能是移动组件的数据错误，也可能是物理系统的计算出现问题，还可能是输入系统传递的指令有误。开发者需要逐一检查相关的组件数据和系统逻辑，这比在面向对象架构中直接跟踪一个对象的方法调用要复杂得多，大大增加了调试的时间和难度。
- **可能存在缓存利用率低的问题**。在 ECS 中，组件通常是按照类型进行存储的，系统在处理实体时，需要遍历具有特定组件组合的实体集合。如果实体的组件分布较为零散，或者系统需要频繁访问不同类型的组件，就可能导致 CPU 缓存命中率降低。因为 CPU 缓存是基于局部性原理工作的，当数据在内存中不连续时，缓存加载会变得频繁且低效，从而影响系统的运行性能。尤其是在大型项目中，实体和组件数量众多，这种缓存问题可能会更加明显，需要开发者进行大量的优化工作才能缓解。
- **对复杂行为的支持不足**。虽然 ECS 架构在处理简单的实体行为时非常高效，但对于一些复杂的行为逻辑，可能需要设计多个系统来协同工作，这会导致系统之间的依赖关系变得复杂。比如，一个角色的攻击行为可能涉及到输入系统、动画系统、碰撞检测系统等多个系统的配合，这种多系统协作的方式在设计和实现上都比较繁琐，容易引入错误。
- **对小型项目来说可能过于复杂**。对于一些功能简单、实体和逻辑较少的小型项目，ECS 架构可能显得过于重量级。采用 ECS 需要设计大量的组件和系统，搭建相应的框架结构，这会增加不必要的开发工作量。相比之下，传统的面向对象架构能够以更简洁的方式实现项目需求，开发效率更高，因此在小型项目中，ECS 的优势难以体现，反而会暴露其复杂性的缺点。

## 3 实战
&emsp;&emsp;用ECS实现2D场景下物体移动效果，效果如下图所示，下图中红色小球是通过键盘控制移动的，其他物体是自由下落循环播放。
![](https://cdn.jsdelivr.net/gh/grayondream/MyImageBlob@main/imgs/GIF%202025-7-19%2014-23-20.gif)

**实体**
&emsp;&emsp;ECS中的Entity只是一个标记记号，用来标识不同的实体，实际中的具体实例是Entity标记和Component共同组成的。
```cpp
using Entity = std::uint32_t;
constexpr static inline std::size_t kMaxEntities = 4096;

using ComponentType = std::uint8_t;
constexpr static inline ComponentType kMaxComponents = 128;
using Signature = std::bitset<kMaxComponents>;
```

**组件**
&emsp;&emsp;组件就是具体的属性，通常只是一个简单的数据结构。我们的场景中要处理物体的移动，键盘响应等事件，那么对应的组件就有：
- Player 组件: 存储玩家的状态，场景比较简单，因此只创建一个空组件用来标识玩家。
```cpp
struct Player{};
```
- Transform 组件: 存储实体的位置、旋转和缩放。
```cpp
struct Transform{
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 scale;
};
```
- Velocity 组件: 存储实体的速度。
```cpp
struct Velocity {
    glm::vec3 speed;
};
```
- Keyboard组件：存储玩家的键盘输入状态。键盘组件可有可无，根据场景需要来定，也可通过其他方式实现。
```cpp
using KeyType = unsigned int;
struct KeyBoard {
    //simply a key code
    KeyType keyCode;
};
```
- Render组件：存储实体的渲染信息。当前场景只需要处理颜色相关数据。
```cpp
struct RenderColor{
    float r;
    float g;
    float b;
    float a;
};
```

**系统**
&emsp;&emsp;系统就是处理具体事件的地方，和具体的事务强绑定。根据目标简单的拆解，相关的系统有：
- 窗口管理系统 (WindowSystem):负责创建和管理窗口。
- 键盘输入系统 (KeyboardSystem):处理键盘输入，控制玩家（红色小球）的移动。
- 渲染系统 (RenderSystem):负责渲染所有实体，包括玩家和自由下落的物体。
- 玩家控制系统 (PlayerControlSystem):处理红色小球的移动逻辑。
- 自由下落系统 (FallingObjectSystem):处理其他物体的下落逻辑。

&emsp;&emsp;这里的实现只是简单的利用set不考虑内存优化。
```cpp
class SystemStoratge{
protected:
    std::set<Entity> _entities;
};

class ISystem{
public:
    virtual void init() = 0;
    virtual void tick(float dt) = 0;
    virtual ~ISystem() = default;
    virtual bool isRunning() const = 0;
    virtual void destroy() = 0;
    virtual void erase(Entity entity) = 0;
    virtual void insert(Entity entity) = 0;
};

class System : public SystemStoratge, public ISystem{
public:
    virtual bool isRunning() const {
        return true;
    }

    virtual void init() override {}
    virtual void tick(float dt) override {}
    virtual void destroy() override {}
    virtual void erase(Entity entity) override {
        _entities.erase(entity);
    }

    virtual void insert(Entity entity) override {
        _entities.insert(entity);
    }
    
};
```

&emsp;&emsp;系统具体实现中，通常会遍历所有实体，根据实体的组件来判断是否需要处理。
```cpp
void PlayerSystem::tick(float dt) {
    for(const auto& entity : _entities) {
        try {
            auto& player = g_worldManager->getComponent<Player>(entity);
            auto& transform = g_worldManager->getComponent<Transform>(entity);
            auto& keyboard = g_worldManager->getComponent<KeyBoard>(entity);
            if(keyboard.keyCode != 0){
                if (keyboard.keyCode == GLFW_KEY_W) {
                    transform.position.y += 0.1f; // Move forward
                } else if (keyboard.keyCode == GLFW_KEY_S) {
                    transform.position.y -= 0.1f; // Move backward
                } else if (keyboard.keyCode == GLFW_KEY_A) {
                    transform.position.x -= 0.1f; // Move left
                } else if (keyboard.keyCode == GLFW_KEY_D) {
                    transform.position.x += 0.1f; // Move right
                }

                LOGI("Player {} moved to position[{}, {}, {}]", (int)entity, transform.position.x, transform.position.y, transform.position.z);
                keyboard.keyCode = 0; // Reset key code after processing
            }
            
            
            //LOGI("Player {} moved to position[{}, {}, {}]", entity, transform.position.x, transform.position.y, transform.position.z);
        } catch (const std::exception& e) {
            LOGE("Error processing entity {}: {}", (int)entity, e.what());
        }
    }
}
```

**Manager**
&emsp;&emsp;到此为止我们只是定义了ECS架构中最基本的组件，似乎还是没办法很清晰的认识到不同组件的作用。为了方便管理不同System-Entity，我们引入```ComponentManager,EntityManager,SystemManager```来分别管理组件，实体，系统，最后使用```WorldManager```统一管理三者。

&emsp;&emsp;EntityManager 类负责管理实体的生命周期和组件签名。它允许创建和销毁实体，并为每个实体存储其组件的签名，以便于系统识别实体的特性。其中签名是一个bitset用来标识当前Entity包含哪些组件。
```cpp
Entity EntityManager::createEntity() {
    if (!availableEntities.empty()) {
        Entity entity = availableEntities.front();
        availableEntities.pop();
        return entity;
    }

    if (currentEntity >= kMaxEntities) {
        throw std::runtime_error("Maximum number of entities reached.");
    }
    return currentEntity++;
}

void EntityManager::destroyEntity(Entity entity) {
    if (entity >= currentEntity) {
        throw std::runtime_error("Entity does not exist.");
    }
    signatures[entity].reset(); // 清除该实体的签名
    availableEntities.push(entity); // 将实体加入可用池
}

void EntityManager::setSignature(Entity entity, Signature signature) {
    if (entity >= currentEntity) {
        throw std::runtime_error("Entity does not exist.");
    }

    signatures[entity] = signature;
}


Signature EntityManager::getSignature(Entity entity) const {
    if (entity >= currentEntity) {
        throw std::runtime_error("Entity does not exist.");
    }
    
    return signatures[entity];
}
```

&emsp;&emsp;ComponentArray 类负责管理组件数组。用来存储当前组件都有哪些Entity持有，同时保存正反向索引来加速访问。
```cpp
template<typename T>
class ComponentArray : public IComponentArray {
public:
    void insert(Entity entity, T component) {
        assert(_entityToIndexMap.find(entity) == _entityToIndexMap.end() && "Component added to the same entity more than once.");

        // Put new entry at end
        size_t newIndex = _size;
        _entityToIndexMap[entity] = newIndex;
        _indexToEntityMap[newIndex] = entity;
        _componentArray[newIndex] = component;
        ++_size;
    }

    //省略部分代码......
    
private:
    std::array<T, kMaxEntities> _componentArray{};
    std::unordered_map<Entity, size_t> _entityToIndexMap{};
    std::unordered_map<size_t, Entity> _indexToEntityMap{};
    size_t _size{0}; // 当前存储的大小
};
```

&emsp;&emsp;ComponentManager 类负责管理组件类型。用来注册组件类型，同时根据组件类型获取对应的ComponentArray。

```cpp
class ComponentManager {
public:
    template<typename T>
    void registerComponent(){
        static_assert(std::is_standard_layout_v<T>, "Component must be standard layout.");
        static_assert(sizeof(T) > 0, "Component must have size.");

        const char* typeName = typeid(T).name();
        if (_componentTypes.find(typeName) != _componentTypes.end()) {
            throw std::runtime_error("Component type already registered.");
        }

        _componentTypes[typeName] = _componentTypes.size();
        _componentStorages[typeName] = std::make_shared<ComponentArray<T>>();
    }

    template<typename T>
    void add(Entity entity, T component){
        getComponentArray<T>()->insert(entity, component);
    }

    //省略部分代码......


private:
    template<typename T>
	std::shared_ptr<ComponentArray<T>> getComponentArray(){
		const char* typeName = typeid(T).name();

		assert(_componentTypes.find(typeName) != _componentTypes.end() && "Component not registered before use.");
		return std::static_pointer_cast<ComponentArray<T>>(_componentStorages[typeName]);
	}

private:
    std::unordered_map<const char*, ComponentType> _componentTypes;
    std::unordered_map<const char*, std::shared_ptr<IComponentArray>> _componentStorages;
};
```

&emsp;&emsp;SystemManager要简单的多主要就是管理系统。
```cpp
class SystemManager{
public:
    template<typename T, typename... Args>
    std::shared_ptr<T> registerSystem(Args&&... args) {
        const char* typeName = typeid(T).name();
        assert(_systems.find(typeName) == _systems.end() && "Registering system more than once.");
        
        auto system = std::make_shared<T>(std::forward<Args>(args)...);
        _systems.insert({typeName, system});
        return system;
    }

    template<typename T>
    void setSignature(Signature signature) {
        const char* typeName = typeid(T).name();
        assert(_systems.find(typeName) != _systems.end() && "System used before registered.");
        
        _signatures.insert({typeName, signature});
    }

    //省略部分代码...............

private:
    std::unordered_map<const char*, std::shared_ptr<ISystem>> _systems;
    std::unordered_map<const char*, Signature> _signatures; // 系统签名
};
```

&emsp;&emsp;最后是WorldManager，其实现比较简单，只是大多数接口都是调用SystemManager和EntityManager的。
```cpp
class WorldManager {
public:
    WorldManager()
        : _sysManager(std::make_unique<SystemManager>()),
          _entityManager(std::make_unique<EntityManager>()),
          _componentManager(std::make_unique<ComponentManager>()) {}

    ~WorldManager() = default;

    template<typename T, typename... Args>
    std::shared_ptr<T> registerSystem(Args&&... args) {
        return _sysManager->registerSystem<T>(std::forward<Args>(args)...);
    }

    void initializeAll() {
        _sysManager->initializeAll();
    }

    //省略部分代码...............
private:
    std::unique_ptr<SystemManager> _sysManager;
    std::unique_ptr<EntityManager> _entityManager;
    std::unique_ptr<ComponentManager> _componentManager;
};
```

&emsp;&emsp;比较的组件都有了，下一步需要的就是将他们组合起来。首先是需要注册我们即将用到的System和组件。
```cpp
void RegisterComponents() {
    g_worldManager->registerComponent<Player>();
    g_worldManager->registerComponent<Transform>();
    g_worldManager->registerComponent<KeyBoard>();
    g_worldManager->registerComponent<RenderColor>();
    g_worldManager->registerComponent<Velocity>();
}

void RegisterSystems() {
    //省略部分代码................
    auto renderSystem = g_worldManager->registerSystem<RenderSystem>();
    {
        Signature signature;
        signature.set(g_worldManager->getComponentType<Transform>());
        signature.set(g_worldManager->getComponentType<RenderColor>());
        g_worldManager->setSystemSignature<RenderSystem>(signature);
    }

}
```

&emsp;&emsp;然后是创建实例。
```cpp
void CreatePlayerEntity() {
    auto playerSystem = g_worldManager->registerSystem<PlayerSystem>();
    {
        Signature signature;
        signature.set(g_worldManager->getComponentType<Player>());
        signature.set(g_worldManager->getComponentType<Transform>());
        signature.set(g_worldManager->getComponentType<KeyBoard>());
        signature.set(g_worldManager->getComponentType<RenderColor>());
        g_worldManager->setSystemSignature<PlayerSystem>(signature);
    }

    Entity playerEntity = g_worldManager->createEntity();
    g_worldManager->addComponent(playerEntity, Player{});
    g_worldManager->addComponent(playerEntity, Transform{ glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f) });
    g_worldManager->addComponent(playerEntity, KeyBoard{ 0 }); // Initialize with a default key code
    g_worldManager->addComponent(playerEntity, RenderColor{ 1.0f, 0.0f, 0.0f, 1.0f }); // Red color
}
```

&emsp;&emsp;使用时需要根据Entity来获取当前Entity对应的组件并且根据场景进行读取和更新，比如下面渲染系统中读取Player和非Player实体的transform和color进行渲染。
```cpp
void RenderSystem::tick(float dt) {
    // This is where rendering logic would go
    glClearColor(0.1, 0.1, 0.1, 1.0);
    GLint viewport[4];

    // 获取当前视口
    glGetIntegerv(GL_VIEWPORT, viewport);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    for(const auto& entity : _entities) {
        try {
            // Example: Get a component and log its data
            auto& transform = g_worldManager->getComponent<Transform>(entity);
            auto& color = g_worldManager->getComponent<RenderColor>(entity);
            if(g_worldManager->hasComponent<Player>(entity)) {
                //LOGI("Rendering player entity {} at position [{}, {}, {}]", (int)entity, transform.position.x, transform.position.y, transform.position.z);
            }else{
                //LOGI("Rendering entity {} at position [{}, {}, {}]", (int)entity, transform.position.x, transform.position.y, transform.position.z);
            }

            //LOGI("Rendering entity {} at position [{}, {}, {}]", (int)entity, transform.position.x, transform.position.y, transform.position.z);
            {
                glBindVertexArray(_vao);
                auto projection = glm::perspective<float>(glm::radians(_camera.zoom()), viewport[3] * 1.0f/viewport[2], 0.1f, 100.0f);
	
                const auto view = _camera.getViewMatrix();
                
                glm::vec3 pos = glm::vec3(
                    transform.position.x,
                    transform.position.y,
                    1
                );
                //draw light source
                {
                    
                    glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
                    model = glm::translate(model, pos);
                    model = glm::rotate(model, 0.f, glm::vec3(1.0f, 0.f, 0.f));
                    model = glm::scale(model, glm::vec3(0.1, 0.1, 0.1));

                    _glProgram->use();
                    _glProgram->update("projection", projection);
                    _glProgram->update("view", view);
                    _glProgram->update("color", glm::vec4(color.r, color.b, color.b, color.a));
                    _glProgram->update("model", model);
                    glDrawElements(GL_TRIANGLES, _shape.idxSize(), GL_UNSIGNED_INT, 0);
                    glBindVertexArray(0);
                }
            }
        } catch (const std::exception& e) {
            LOGE("Error processing entity {}: {}", (int)entity, e.what());
        }

        
    }

}
```

## 4 参考文献
- [Github-ECSDemo](https://github.com/dereksawicki/Entity-Component-System-Demo/blob/master/Renderer.cpp)
- [entity_component_system](https://austinmorlan.com/posts/entity_component_system/)
- [一文看懂ECS架构](https://zhuanlan.zhihu.com/p/618971664)